# Copyright (c) OpenMMLab. All rights reserved.
# modify from https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/voxel_encoders/voxel_encoder.py
# create by -zyrant

import torch
from torch import nn
from math import sqrt

from mmdet3d.models.builder import VOXEL_ENCODERS
from torch_scatter import scatter_max, scatter_mean
from mmdet3d.ops import knn
from mmdet3d.core.bbox.structures import rotation_3d_in_axis

try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow get_started.md to install MinkowskiEngine.
    ME = None
    pass


class SuperpointAttention_v2(nn.Module):
    def __init__(self, feature_dim, k=8, dropout_rate=0.1):
        super(SuperpointAttention_v2, self).__init__()
        self.k = k
        self.feature_transform = nn.Linear(feature_dim, feature_dim)
        self.coord_linear = nn.Linear(3, feature_dim)
        self.feat_linear = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=1)
        self.norm = nn.LayerNorm(feature_dim)
        self.scale = sqrt(k)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, features, coords):

        """
        :param features: (B*N, C)
        :param coords:  (B*N, 4), first dim: batch_id
        :return:  (B*N, C)
        """

        # attn
        batch_size = (coords[:, 0].max() + 1).int()
        output_features_list = []
        for batch_id in range(batch_size): #按批次处理
            batch_mask = coords[:, 0] == batch_id  # creat batch_mask
            
            batch_features = features[batch_mask]  # (N_i, C)
            batch_coords = coords[batch_mask, 1:]  # (N_i, 3)

            # get knn index 
            indices = knn(self.k, batch_coords[None, ::].contiguous(), 
                                    batch_coords[None, ::].contiguous())[0].squeeze(0).long().transpose(0, 1)
            
            neighbor_coords = batch_coords[indices]  # (N_i, k, 3)
            neighbor_features = batch_features[indices]  # (N_i, k, C)

            relative_coords = neighbor_coords - batch_coords.unsqueeze(1) # (N_i, k, 3)

            relative_feats = neighbor_features - batch_features.unsqueeze(1) # (N_i, k, c) #计算每个点与其邻居之间的相对坐标和特征差值
         
            attention = self.softmax(self.coord_linear(relative_coords) * self.feat_linear(relative_feats) / self.scale)  # (N_i, k, C)

            output_batch_features = torch.sum(self.dropout(attention) * self.feature_transform(neighbor_features), dim=1)  # (N_i, C)
            output_features_list.append(output_batch_features)
        updated_features = torch.cat(output_features_list, dim=0)  # (B*N, C)

        # res
        output = updated_features + features
        output = self.norm(output)

        return output
    

@VOXEL_ENCODERS.register_module()
class CPG_encoder(nn.Module):
    def __init__(self,
                 in_channels=4,
                 voxel_size = 0.02,
                 latter_voxel_size = 0.04,
                 local_k = 8,
                 with_yaw = False,
                 feat_channels=None,
                 with_xyz = False,
                 with_distance=False, # no using
                 with_cluster_center=False, # no using
                 with_superpoint_center=False,
                 mode='max',
                 return_point_feats=False):
        super(CPG_encoder, self).__init__()
        assert mode in ['avg', 'max','attn', 'mean_max']
        assert len(feat_channels) > 0
        self.mode = mode
        self.with_yaw = with_yaw

        # voxel vote  https://github.com/Haiyang-W/CAGroup3D
        self.offset_block = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, in_channels, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(in_channels),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(in_channels, in_channels, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(in_channels),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(in_channels, 3+in_channels, kernel_size=1, dimension=3))
       

        if with_xyz:
            in_channels += 3
        if with_cluster_center:
            in_channels += 3
        if with_superpoint_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self.voxel_size = voxel_size
        self.latter_voxel_size = latter_voxel_size

        self.in_channels = in_channels
        self._with_xyz = with_xyz
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_superpoint_center = with_superpoint_center
        self.return_point_feats = return_point_feats
        self.fp16_enabled = False

        # superpoint-voxel fusion
        feat_channels = [self.in_channels] + list(feat_channels)
        ffn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            in_filters *= 2
            ffn_layers.append(
                nn.Sequential(
                    ME.MinkowskiConvolution(in_filters, out_filters,  kernel_size = 3, dilation = 3, dimension=3), 
                    ME.MinkowskiBatchNorm(out_filters),
                    ME.MinkowskiELU(),
                    ))
        self.ffn_layers = nn.ModuleList(ffn_layers)
        self.num_ffn = len(ffn_layers)

        # superpoint_attention
        attn_layers = []
        for i in range(len(feat_channels)):
            in_filters = feat_channels[i]
            attn_layers.append(
                    SuperpointAttention_v2(in_filters, local_k)
                    )
        self.attn_layers = nn.ModuleList(attn_layers)

        
        self.out_linear = nn.Sequential(nn.Linear(sum(feat_channels), sum(feat_channels), bias=False),
                                        nn.LayerNorm(sum(feat_channels)),
                                        nn.ELU(),
                                        nn.Linear(sum(feat_channels), sum(feat_channels), bias=False),
                                        nn.LayerNorm(sum(feat_channels)),
                                        nn.ELU())
        
#索引查找
    def map_voxel_center_to_point(self, voxel_mean, voxel2point_inds):
        return voxel_mean[voxel2point_inds]
    
    # @staticmethod
    def _get_face_distances(self,points, boxes):
        """Calculate distances from point to box faces.

        Args:
            points (Tensor): Final locations of shape (N_points, N_boxes, 3).
            boxes (Tensor): 3D boxes of shape (N_points, N_boxes, 7)

        Returns:
            Tensor: Face distances of shape (N_points, N_boxes, 6),
                (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).
        """
        shift = torch.stack(
            (points[..., 0] - boxes[..., 0], points[..., 1] - boxes[..., 1],
             points[..., 2] - boxes[..., 2]),
            dim=-1).permute(1, 0, 2)
        #scannet没有旋转
        shift = rotation_3d_in_axis(
            shift, -boxes[0, :, 6], axis=2).permute(1, 0, 2)
        centers = boxes[..., :3] + shift
        dx_min = centers[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
        dx_max = boxes[..., 0] + boxes[..., 3] / 2 - centers[..., 0]
        dy_min = centers[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
        dy_max = boxes[..., 1] + boxes[..., 4] / 2 - centers[..., 1]
        dz_min = centers[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
        dz_max = boxes[..., 2] + boxes[..., 5] / 2 - centers[..., 2]
        return torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max),
                           dim=-1)
        
    
               
    
    def forward(self,
                x,
                points,
                superpoints,
                img_metas=None,
                gt_bboxes_3d=None,
                *args,
                **kwargs):
        
        #--Geometry-Aware Voting--
        scene_coord = x.C[:, 1:].clone()
        max_bound = (scene_coord.max(0)[0] + x.coordinate_map_key.get_key()[0][0]) * self.voxel_size
        min_bound = (scene_coord.min(0)[0] - x.coordinate_map_key.get_key()[0][0]) * self.voxel_size
        
        
        vote_x = self.offset_block(x)

        vote_coordinates = x.coordinates[:, 1:].float().clone() * self.voxel_size + vote_x.features[:, :3].clone().detach() # backbone coordinates + offset coordinates
        vote_coordinates[:, 0] = torch.clamp(vote_coordinates[:, 0], max=max_bound[0], min=min_bound[0])
        vote_coordinates[:, 1] = torch.clamp(vote_coordinates[:, 1], max=max_bound[1], min=min_bound[1])
        vote_coordinates[:, 2] = torch.clamp(vote_coordinates[:, 2], max=max_bound[2], min=min_bound[2])

        orgin_coordinates = x.coordinates[:, 1:].float().clone() * self.voxel_size
        vote_feats = x.features + vote_x.features[:, 3:]
        vote_feats_norm = torch.norm(vote_feats, p=2, dim=1)
        vote_feats = vote_feats.div(vote_feats_norm.unsqueeze(1))

        features_ms, coordinates_ms, coordinates_offsets, vote_offsets, vote_voxel_points = [], [], [], [], []
        for permutation in x.decomposition_permutations:
            batch_features = torch.cat([x.features[permutation], vote_feats[permutation]], dim=0) # merge
            batch_coordinates = torch.cat([orgin_coordinates[permutation], vote_coordinates[permutation]], dim=0)
            batch_coordinates_offsets = x.coordinates[:, 0][permutation].repeat(2) 
            vote_offsets.append(vote_x.features[:,:3][permutation])
            vote_voxel_points.append(vote_x.coordinates[permutation][:, 1:] * self.voxel_size) 
            features_ms.append(batch_features) 
            coordinates_ms.append(batch_coordinates) 
            coordinates_offsets.append(batch_coordinates_offsets) #[[0,0...0],...,[3,3,3..3]]

    
        
        
        # --get unique superpoint id and batch_offsets--
        unique_superpoints, batch_offsets, num_superpoints, orgin_superpoints = [], [], [], []
        count = 0
        for batch_ids in range(len(points)):
          
            point_idx = knn(1, points[batch_ids][None, :, :3].contiguous(), 
                                    vote_voxel_points[batch_ids][None, ::].contiguous())[0].squeeze(0) # remember to use coordinates before vote
           
            batch_select_superpoints = superpoints[batch_ids][point_idx.long()]
            sp_ids, batch_select_superpoints = torch.unique(batch_select_superpoints, return_inverse=True) #batch_select_superpoints长度与体素一样，代表每个体素属于哪个超级点id,这个超点是根据原始体素选出来的。结果是一个索引
            
            orgin_superpoints.append(batch_select_superpoints)
            batch_unique_superpoints = count + batch_select_superpoints
            batch_unique_superpoints = batch_unique_superpoints.repeat(2)
            count += (batch_select_superpoints.max() + 1)
            unique_superpoints.append(batch_unique_superpoints)
            repeat_num = (batch_select_superpoints.max() + 1) 
            num_superpoints.append(repeat_num)
            batch_index = torch.tensor(batch_ids, device = batch_select_superpoints.device).repeat(repeat_num).unsqueeze(-1)
            batch_offsets.append(batch_index)
        unique_superpoints = torch.cat(unique_superpoints, dim=0).squeeze(-1) 
        batch_offsets = torch.cat(batch_offsets, dim=0)
    
        

        # --iter superpoint attention and Superpoint-Voxel Fusion--
        features = torch.cat(features_ms, dim=0)
        coordinates = torch.cat(coordinates_ms, dim=0)
        coordinates_offsets = torch.cat(coordinates_offsets, dim=0)
        

        features_ls = [features]
        superpoint_mean_center = scatter_mean(coordinates, unique_superpoints, dim=0)
        if self._with_xyz:
            features_ls.append(coordinates) 

        # Find distance of x, y, and z from superpoint center
        if self._with_superpoint_center:
            points_mean_center = self.map_voxel_center_to_point(superpoint_mean_center, unique_superpoints) #看看每个体素都属于哪个superpoint_mean_center（超点）,长度是体素总数量(如果把sp剪枝，那有的体素就会没有sp中心)
            f_superpoint_center = coordinates - points_mean_center 
            features_ls.append(f_superpoint_center)  # xyz-superpoint center

        if self._with_distance: # not implement
            raise NotImplementedError
        
        if self._with_cluster_center: # not implement
            raise NotImplementedError


        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1) # feats，xyzs, xyz-superpoint center (NxD) D=(64+3+3),N=Mx2
        voxel_coods = torch.cat((batch_offsets, superpoint_mean_center), dim=1) 
        point_feats = features #每个体素的特征
        
        voxel_feats_list = []
        for i, attn in enumerate(self.attn_layers):

            #--superpoint pooling--
            if self.mode == 'max': 
                voxel_feats, _ = scatter_max(point_feats, unique_superpoints, dim=0)
            elif self.mode == 'avg':
                voxel_feats = scatter_mean(point_feats, unique_superpoints, dim=0)
            else:
                raise NotImplementedError
        
            
            #--superpoint Attention--
            voxel_feats = attn(voxel_feats, voxel_coods)
            voxel_feats_list.append(voxel_feats)

            if i != len(self.attn_layers) - 1:

                #--Superpoint-Voxel Fusion--
                feat_per_point = self.map_voxel_center_to_point(
                    voxel_feats, unique_superpoints) 
                features = torch.cat([point_feats, feat_per_point], dim=1) #g
                
                sparse_coordinates = torch.cat([coordinates_offsets.unsqueeze(-1), (coordinates / self.latter_voxel_size).floor()], dim=1) 
                sparse_features = ME.SparseTensor(coordinates = sparse_coordinates, 
                                            features = features) 

                point_feats = self.ffn_layers[i](sparse_features)
                point_feats = point_feats.slice(sparse_features).features 

        voxel_feats = torch.cat(voxel_feats_list, dim=1) #len(voxel_feats[0])=64+70+128+128=390
        voxel_feats = self.out_linear(voxel_feats) + voxel_feats

        #sp feats
        voxel_coordinates = voxel_coods.clone()
        voxel_coordinates[:, 1:] = torch.floor(voxel_coordinates[:, 1:] / self.voxel_size)
        decode_out = ME.SparseTensor(features=voxel_feats, coordinates=voxel_coordinates.int().to(voxel_feats.device))
       
        
        feats_dict = dict(x = x, #feat afer backbone 41893
                           voxel_feats = voxel_feats, #1336 
                            voxel_coods = voxel_coods, 
                            vote_offsets = vote_offsets,  #41893
                            vote_voxel_points = vote_voxel_points,
                            orgin_superpoints = orgin_superpoints,
                            decode_out = decode_out) 
        return feats_dict
