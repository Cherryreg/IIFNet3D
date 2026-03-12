try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')
import torch.nn.functional as F
import torch
from mmcv.cnn import bias_init_with_prob
from mmcv.runner import BaseModule
from torch import nn
from .cagroup_proposal_target_layer import ProposalTargetLayer
from mmdet3d.models.builder import HEADS, build_loss
from . import common_utils
import numpy as np
#from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from .cagroup_utils import CAGroupResidualCoder as ResidualCoder
#from mmdet3d.utils.iou3d_loss import IoU3DLoss
from .loss_utils import WeightedSmoothL1Loss
from mmcv.ops import nms3d, nms3d_normal
from mmcv.utils import Registry
import time
import torch.nn as nn

    
    
class PositionEncoding_3d_q(nn.Module):
    def __init__(self, pos_dim=7, feature_dim=128, hidden_dim=64):
        super().__init__()
        
        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.pos_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, roi_center):
        pos_embed = self.pos_mlp(roi_center)  # [B*N, C]
        # pooled_features_list = [f + pos_embed for f in pooled_features_list]  # Broadcast 加法
        return pos_embed

class PositionEncoding_3d_k(nn.Module):
    def __init__(self, pos_dim=7, feature_dim=128, hidden_dim=64):
        super().__init__()
        
        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.pos_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, roi_center):
        pos_embed = self.pos_mlp(roi_center)  # [B*N, C]
        # pooled_features_list = [f + pos_embed for f in pooled_features_list]  # Broadcast 加法
        return pos_embed


class PositionEncoding_2d(nn.Module):
    def __init__(self, pos_dim=3, feature_dim=128, hidden_dim=64):
        super().__init__()
        
        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.pos_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, roi_center):
        pos_embed = self.pos_mlp(roi_center)  # [B*N, C]
        return pos_embed  

class BiCrossAttentionLayer_v2(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.attn_3d_to_2d = nn.MultiheadAttention(embed_dim=dim, num_heads=1, dropout=dropout, batch_first=True)
        self.attn_2d_to_3d = nn.MultiheadAttention(embed_dim=dim, num_heads=1, dropout=dropout, batch_first=True)
        self.norm_3d = nn.LayerNorm(dim)
        self.norm_2d = nn.LayerNorm(dim)

        self.samples = 0
        self.attention_3d_time = 0.0
        self.attention_2d_time = 0.0


    def forward(self, feat_3d, feat_2d, pos_embed_3d_q,pos_embed_3d_k,pos_embed_2d, B, N):
     
        feat_3d = feat_3d.view(B, N, -1)  # [B, N, C]
        feat_2d = feat_2d.view(B, N, -1)  # [B, N, C]
        
        pos_embed_3d_q = pos_embed_3d_q.view(B, N, -1)  # [B, N, C]
        pos_embed_3d_k = pos_embed_3d_k.view(B, N, -1)  # [B, N, C]
        
        
        pos_embed_2d = pos_embed_2d.view(B, N, -1)  # [B, N, C]  
        
        q_3d = feat_3d + pos_embed_3d_q  # [B, N, C]
        kv_2d = feat_2d + pos_embed_3d_k  # [B, N, C]

        fused_3d, _ = self.attn_3d_to_2d(q_3d, kv_2d, kv_2d)
        
        fused_3d = self.norm_3d(fused_3d + q_3d)  # Residual

        
        q_2d = feat_2d + pos_embed_2d  # [B, N, C]
        kv_3d = feat_3d + pos_embed_2d  # [B, N, C]

   
        fused_2d, _ = self.attn_2d_to_3d(q_2d, kv_3d, kv_3d)  # [B, N, C]
   
      
        fused_2d = self.norm_2d(fused_2d + q_2d)  # Residual


        fused = fused_3d + fused_2d

        
        return fused.reshape(B * N, -1)                     
                        
                         
class CalWeight3D(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super(CalWeight3D, self).__init__()
        
        self.mod_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # self.mod_mlp = nn.Sequential(
        #     nn.Linear(feature_dim*2, feature_dim),
        #     # nn.ReLU(),
        #     # nn.Linear(hidden_dim, 1),
        #     nn.Sigmoid()
        # )
      
        
        self.init_weights()

    def forward(self, feats):
        """
        Args:
            feat: Tensor of shape [B*N, C]
        
        Returns:
            weights: Tensor of shape [B*N, 2]
        """
        w1 = self.mod_mlp(feats)  # [B*N, 1]
        w2 = 1-w1
    
        weights = torch.cat([w1, w2], dim=1)  # [B*N, 2]
     
        return weights
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                       nn.init.constant_(m.bias, 0)

class GatedFusion(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Sigmoid()
        )

    def forward(self, feat1, feat2):
        gate = self.gate(torch.cat([feat1, feat2], dim=-1))
        return gate * feat1 + (1 - gate) * feat2 
                





class SimplePoolingLayer(nn.Module):
    def __init__(self, channels=[128,128,128], grid_kernel_size = 5, grid_num = 7, voxel_size=0.04, coord_key=2,
                    point_cloud_range=[-5.12*3, -5.12*3, -5.12*3, 5.12*3, 5.12*3, 5.12*3], # simply use a large range
                    corner_offset_emb=False, pooling=False):
        super(SimplePoolingLayer, self).__init__()
        # build conv
        self.voxel_size = voxel_size
        self.coord_key = coord_key
        grid_size = [int((point_cloud_range[3] - point_cloud_range[0])/voxel_size), 
                     int((point_cloud_range[4] - point_cloud_range[1])/voxel_size), 
                     int((point_cloud_range[5] - point_cloud_range[2])/voxel_size)]
        self.grid_size = grid_size
        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]
        self.grid_num = grid_num
        self.pooling = pooling
        self.count = 0
        self.grid_conv = ME.MinkowskiConvolution(channels[0], channels[1], kernel_size=grid_kernel_size, dimension=3)
        self.grid_bn = ME.MinkowskiBatchNorm(channels[1])
        self.grid_relu = ME.MinkowskiELU()
        if self.pooling:
            self.pooling_conv = ME.MinkowskiConvolution(channels[1], channels[2], kernel_size=grid_num, dimension=3)
            self.pooling_bn = ME.MinkowskiBatchNorm(channels[1])

        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(self.grid_conv.kernel, std=.01)
        if self.pooling:
            nn.init.normal_(self.pooling_conv.kernel, std=.01)

    def forward(self, sp_tensor, grid_points, grid_corners=None, box_centers=None, batch_size=None):
        """
        Args:
            sp_tensor: minkowski tensor
            grid_points: bxnum_roisx216, 4 (b,x,y,z)
            grid_corners (optional): bxnum_roisx216, 8, 3
            box_centers: bxnum_rois, 4 (b,x,y,z)
        """
        grid_coords = grid_points.long()
        grid_coords[:, 1:4] = torch.floor(grid_points[:, 1:4] / self.voxel_size) # get coords (grid conv center)
        grid_coords[:, 1:4] = torch.clamp(grid_coords[:, 1:4], min=-self.grid_size[0] / 2 + 1, max=self.grid_size[0] / 2 - 1) # -192 ~ 192
        grid_coords_positive = grid_coords[:, 1:4] + self.grid_size[0] // 2 
        merge_coords = grid_coords[:, 0] * self.scale_xyz + \
                        grid_coords_positive[:, 0] * self.scale_yz + \
                        grid_coords_positive[:, 1] * self.scale_z + \
                        grid_coords_positive[:, 2] 
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)
        unq_grid_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1) 
        unq_grid_coords[:, 1:4] -= self.grid_size[0] // 2
        unq_grid_coords[:, 1:4] *= self.coord_key
        unq_grid_sp_tensor = self.grid_relu(self.grid_bn(self.grid_conv(sp_tensor, unq_grid_coords.int()))) 
        unq_features = unq_grid_sp_tensor.F
        unq_coords = unq_grid_sp_tensor.C #numroi*7*7*7
        new_features = unq_features[unq_inv]

        if self.pooling:
            # fake grid
            fake_grid_coords = torch.ones(self.grid_num, self.grid_num, self.grid_num, device=unq_grid_coords.device)
            fake_grid_coords = torch.nonzero(fake_grid_coords) - self.grid_num // 2 
            fake_grid_coords = fake_grid_coords.unsqueeze(0).repeat(grid_coords.shape[0] // fake_grid_coords.shape[0], 1, 1) 
            # fake center
            fake_centers = fake_grid_coords.new_zeros(fake_grid_coords.shape[0], 3) 
            fake_batch_idx = torch.arange(fake_grid_coords.shape[0]).to(fake_grid_coords.device) 
            fake_center_idx = fake_batch_idx.reshape([-1, 1])
            fake_center_coords = torch.cat([fake_center_idx, fake_centers], dim=-1).int() #每个roi的中心点
            
            fake_grid_idx = fake_batch_idx.reshape([-1, 1, 1]).repeat(1, fake_grid_coords.shape[1], 1) 
            fake_grid_coords = torch.cat([fake_grid_idx, fake_grid_coords], dim=-1).reshape([-1, 4]).int()

            grid_sp_tensor = ME.SparseTensor(coordinates=fake_grid_coords, features=new_features)
            pooled_sp_tensor = self.pooling_conv(grid_sp_tensor, fake_center_coords) 
            pooled_sp_tensor = self.pooling_bn(pooled_sp_tensor) 
            return pooled_sp_tensor.F
        else:
            return new_features
        
@HEADS.register_module()
class IIFROIHead(BaseModule):
    def __init__(self,n_classes,grid_size,middle_feature_source,voxel_size,coord_key,mlps,code_size,encode_sincos,roi_per_image,
                 roi_fg_ratio,reg_fg_thresh,roi_conv_kernel,enlarge_ratio,use_iou_loss,use_grid_offset,use_simple_pooling,use_center_pooling,
                 loss_weights,
                #  cls_loss_type='BinaryCrossEntropy',
                 iou_loss = dict(type='AxisAlignedIoULoss', reduction='none'),
                 cls_loss=dict(type='FocalLoss', reduction='none'),
                 reg_loss_type='smooth-l1',
                #  criterion=None,
                 train_cfg=None,
                 test_cfg=None):
        super(IIFROIHead, self).__init__()
        self.middle_feature_source = middle_feature_source # default [3] : only use semantic feature from backbone3d
        self.num_class = n_classes
        self.code_size = code_size
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.enlarge_ratio = enlarge_ratio
        self.mlps = mlps
        self.shared_fc = [256,256]
       
        
        self.cls_fc = [256,256]
        self.reg_fc = [256,256]
        
    
        self.reg_loss_type = reg_loss_type
        self.count = 0
        
        dp_ratio=0.3
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.cls_loss = build_loss(cls_loss)
        
        



        self.encode_angle_by_sincos = encode_sincos
        self.use_iou_loss = use_iou_loss
        if self.use_iou_loss:
            # print('hehe')
            self.iou_loss_computer = build_loss(iou_loss)
            #self.iou_loss_computer = IoU3DLoss(loss_weight=1.0, with_yaw=self.code_size > 6)
        self.use_grid_offset = use_grid_offset
        self.use_simple_pooling = use_simple_pooling
        self.use_center_pooling = use_center_pooling

        self.loss_weight = loss_weights
        self.proposal_target_layer = ProposalTargetLayer(roi_per_image=roi_per_image, 
                                                         fg_ratio=roi_fg_ratio, 
                                                         reg_fg_thresh=reg_fg_thresh,n_classes=self.num_class)
        self.box_coder = ResidualCoder(code_size=code_size, encode_angle_by_sincos=self.encode_angle_by_sincos)
        self.reg_loss_func = WeightedSmoothL1Loss(code_weights=self.loss_weight.code_weight)

        self.roi_grid_pool_layers = nn.ModuleList()
        for i in range(len(self.mlps)): # different feature source, default only use semantic feature
            mlp = self.mlps[i] 
            pool_layer = SimplePoolingLayer(channels=mlp, grid_kernel_size=roi_conv_kernel, grid_num=grid_size, \
                                            voxel_size=voxel_size*coord_key, coord_key=coord_key, pooling=self.use_center_pooling)
            self.roi_grid_pool_layers.append(pool_layer)
        
        if self.use_center_pooling: 
            # pre_channel = sum([x[-1] for x in self.mlps])-self.mlps[0][-1] #256
            #pre_channel = self.mlps[0][-1] # add center position
            pre_channel = 128 # add center position
            # weight_channel = sum([x[-1] for x in self.mlps])
            weight_channel = 256
            # attn_channel = 256
        else:
            raise NotImplementedError
        
        self.cal_weight_3d = CalWeight3D(feature_dim=weight_channel ,hidden_dim=weight_channel//2)
        self.position_encoding_3d_q = PositionEncoding_3d_q(pos_dim=7, feature_dim=128, hidden_dim=64)
        self.position_encoding_3d_k = PositionEncoding_3d_k(pos_dim=7, feature_dim=128, hidden_dim=64)
        self.position_encoding_2d = PositionEncoding_2d(pos_dim=3, feature_dim=128, hidden_dim=64)
        self.cross_attention = BiCrossAttentionLayer_v2(dim=pre_channel, dropout=0.1)

        reg_fc_list = []
        for k in range(0, self.reg_fc.__len__()):
            reg_fc_list.extend([
                nn.Linear(pre_channel, self.reg_fc[k], bias=False),
                nn.BatchNorm1d(self.reg_fc[k]),
                nn.ReLU()
                # ChannelAttention(channels=self.reg_fc[k])
            ])
            pre_channel = self.reg_fc[k]

            if k != self.reg_fc.__len__() - 1 and dp_ratio > 0:
                reg_fc_list.append(nn.Dropout(dp_ratio))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)

        if self.encode_angle_by_sincos:
            self.reg_pred_layer = nn.Linear(pre_channel, self.code_size+1, bias=True)
        else:
            self.reg_pred_layer = nn.Linear(pre_channel, self.code_size, bias=True)
        
        
        if self.use_center_pooling:
            # pre_channel = sum([x[-1] for x in self.mlps])
            # pre_channel = self.mlps[0][-1] #shortcut
             pre_channel = 128 # add center position  
        else:
            raise NotImplementedError
        
        
        # out_channel = self.common_channel
        #added cls-layer
        cls_fc_list = []
        for k in range(0, self.cls_fc.__len__()):
            cls_fc_list.extend([
                nn.Linear(pre_channel, self.cls_fc[k], bias=False),
                nn.BatchNorm1d(self.cls_fc[k]),
                nn.ReLU()
                # ChannelAttention(channels=self.cls_fc[k])
            ])
            pre_channel = self.cls_fc[k]

            if k != self.cls_fc.__len__() - 1 and dp_ratio > 0:
                cls_fc_list.append(nn.Dropout(dp_ratio))
        self.cls_fc_layers = nn.Sequential(*cls_fc_list)

        
        self.cls_pred_layer = nn.Linear(pre_channel, self.num_class, bias=True)
     
     
        
    def init_weights(self): 
        init_func = nn.init.xavier_normal_
        layers_list = [self.shared_fc_layer, self.reg_fc_layers] if not self.use_center_pooling else [self.reg_fc_layers,self.cls_fc_layers]
        for module_list in layers_list:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.normal_(self.cls_pred_layer.weight, mean=0, std=0.01)
        nn.init.constant_(self.reg_pred_layer.bias, 0)
        nn.init.constant_(self.cls_pred_layer.bias, bias_init_with_prob(.01))
        
    
    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1]) 
        batch_size_rcnn = rois.shape[0]
        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size) 
        if self.code_size > 6:
            # global_roi_grid_points =  rotation_3d_in_axis(#!!!!!!!!
            #     local_roi_grid_points.clone(), rois[:, 6],axis=2
            # ).squeeze(dim=1)
            global_roi_grid_points = common_utils.rotate_points_along_z(#!!!!!!!!
                local_roi_grid_points.clone(), rois[:, 6]
            ).squeeze(dim=1)
        else:
            global_roi_grid_points = local_roi_grid_points

        global_center = rois[:, 0:3].clone()
        global_roi_grid_points = global_roi_grid_points + global_center.unsqueeze(dim=1) 
        return global_roi_grid_points, local_roi_grid_points
    
    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  
        
        return roi_grid_points 
    
    def roi_grid_pool(self, input_dict):
        """
        Args:
            input_dict:
                rois: b, num_max_rois, 7
                batch_size: b
                middle_feature_list: List[mink_tensor]
        """
        rois = input_dict['rois']
        batch_size = input_dict['batch_size']
        middle_feature_list = [input_dict['middle_feature_list'][i] for i in self.middle_feature_source]
        if not isinstance(middle_feature_list, list):
            middle_feature_list = [middle_feature_list]
        
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.grid_size
        )  
        
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)

        batch_idx = rois.new_zeros(batch_size, roi_grid_xyz.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        
        pooled_features_list = []
        for k, cur_sp_tensors in enumerate(middle_feature_list):
            pool_layer = self.roi_grid_pool_layers[k]
            if self.use_simple_pooling:
                batch_grid_points = torch.cat([batch_idx, roi_grid_xyz], dim=-1) 
                batch_grid_points = batch_grid_points.reshape([-1, 4])
                new_features = pool_layer(cur_sp_tensors, grid_points=batch_grid_points)
            else:
                raise NotImplementedError
            pooled_features_list.append(new_features)
        
        #get roi centers/whl
        roi_centers = rois[:, :, 0:3].clone()  # [B, N, 3]
        

        rois_flat = rois.view(-1, 7)
        centers = rois_flat[:, 0:6].clone()
        dims = rois_flat[:, 3:6].clone()
        volume = (dims[:, 0] * dims[:, 1] * dims[:, 2]).unsqueeze(1)
        pos = torch.cat([centers, volume], dim=1)
        pos = torch.sin(pos)
       
        ms_pooled_feature_3d = torch.cat([pooled_features_list[0], pooled_features_list[2]], dim=-1)  # [B*N, C]
        weights_3d = self.cal_weight_3d(ms_pooled_feature_3d)  # [B*N, 2]
        stacked_feats_3d = torch.stack([pooled_features_list[0], pooled_features_list[2]], dim=1)  # [B*N, 2, C]
        fused_3d_feat= torch.sum(weights_3d.unsqueeze(-1) * stacked_feats_3d, dim=1)  # [B*N, C]
        pos_embed_3d_q = self.position_encoding_3d_q(pos)  # [B*N, C]
        pos_embed_3d_k = self.position_encoding_3d_k(pos)
        pos_embed_2d = self.position_encoding_2d(roi_centers) 
        rgb_feat = pooled_features_list[1] # [B*N, C]

        
        ms_pooled_feature = self.cross_attention(fused_3d_feat,rgb_feat,pos_embed_3d_q,pos_embed_3d_k,pos_embed_2d,batch_size, rois.shape[1]) #3d is q

        return ms_pooled_feature
    
    #for train no nms in stage one 
    def reorder_rois_for_refining_v1(self,pred_boxes_3d):
        batch_size = len(pred_boxes_3d)
        num_max_rois = max([len(preds) for preds in pred_boxes_3d])
        num_max_rois = max(1, num_max_rois)
        pred_boxes = pred_boxes_3d[0]
        
        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        # rois_assigned_ids = pred_boxes.new_zeros((batch_size, num_max_rois))
        for bs_idx in range(batch_size):
             num_boxes = len(pred_boxes_3d[bs_idx])
             rois[bs_idx, :num_boxes, :] = pred_boxes_3d[bs_idx]
            #  rois_assigned_ids[bs_idx,:num_boxes] = pred_boxes_3d[bs_idx][1]
             
        # converse heading to pcdet heading！
        rois[..., 6] *= -1
        
        return rois
    
    #for stage-one after nms   
    def reoder_rois_for_refining(self, pred_boxes_3d):
        """
        Args:
            pred_boxes_3d: List[(box, score, label), (), ...]
        """
        batch_size = len(pred_boxes_3d)
        num_max_rois = max([len(preds[0]) for preds in pred_boxes_3d])
        num_max_rois = max(1, num_max_rois)
        pred_boxes = pred_boxes_3d[0][0]

        if len(pred_boxes_3d[0]) == 5:
            use_sem_score = True
        else:
            use_sem_score = False

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()
        roi_centerness = pred_boxes.new_zeros((batch_size, num_max_rois))
        if use_sem_score:
            roi_sem_scores = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes_3d[0][3].shape[-1]))

        for bs_idx in range(batch_size):
            num_boxes = len(pred_boxes_3d[bs_idx][0])            
            rois[bs_idx, :num_boxes, :] = pred_boxes_3d[bs_idx][0]
            roi_scores[bs_idx, :num_boxes] = pred_boxes_3d[bs_idx][1]
            roi_labels[bs_idx, :num_boxes] = pred_boxes_3d[bs_idx][2]
            roi_centerness[bs_idx, :num_boxes] = pred_boxes_3d[bs_idx][3].squeeze(-1)
            
            
            #added for assign
            # if len(pred_boxes_3d[0]) == 4:
            #     roi_stage_one_assigned_ids[bs_idx,:num_boxes] = pred_boxes_3d[bs_idx][3]
            if use_sem_score:
                roi_sem_scores[bs_idx, :num_boxes] = pred_boxes_3d[bs_idx][3]
        
        # converse heading to pcdet heading！
        rois[..., 6] *= -1
        if use_sem_score:
            return rois, roi_scores, roi_labels, roi_sem_scores, batch_size,roi_centerness
        # else:
        # if len(pred_boxes_3d[0]) == 4:
        #     return rois, roi_scores, roi_labels, batch_size,roi_stage_one_assigned_ids
        else:
            return rois, roi_scores, roi_labels, batch_size,roi_centerness
    #no cls score in stage-one
    def reoder_rois_for_refining_test_v1(self, pred_boxes_3d):
        """
        Args:
            pred_boxes_3d: List[(box, score, label), (), ...]
        """
        batch_size = len(pred_boxes_3d)
        num_max_rois = max([len(preds[0]) for preds in pred_boxes_3d])
        num_max_rois = max(1, num_max_rois)
        pred_boxes = pred_boxes_3d[0][0]

        if len(pred_boxes_3d[0]) == 5:
            use_sem_score = True
        else:
            use_sem_score = False

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
       
        roi_centerness = pred_boxes.new_zeros((batch_size, num_max_rois))
        #add for assign
        # if len(pred_boxes_3d[0]) == 4:
        #     roi_stage_one_assigned_ids = pred_boxes.new_zeros((batch_size, num_max_rois))
        if use_sem_score:
            roi_sem_scores = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes_3d[0][3].shape[-1]))

        for bs_idx in range(batch_size):
            num_boxes = len(pred_boxes_3d[bs_idx][0])            
            rois[bs_idx, :num_boxes, :] = pred_boxes_3d[bs_idx][0]
    
            roi_centerness[bs_idx, :num_boxes] = pred_boxes_3d[bs_idx][1].squeeze(-1)
    
            if use_sem_score:
                roi_sem_scores[bs_idx, :num_boxes] = pred_boxes_3d[bs_idx][2]
        
        # converse heading to pcdet heading！
        rois[..., 6] *= -1
        if use_sem_score:
            return rois, roi_sem_scores, batch_size
        # else:
        # if len(pred_boxes_3d[0]) == 4:
        #     return rois, roi_scores, roi_labels, batch_size,roi_stage_one_assigned_ids
        else:
            return rois,batch_size,roi_centerness
    
    #no nms score is a vector
    def reoder_rois_for_refining_test(self, pred_boxes_3d):
        """
        Args:
            pred_boxes_3d: List[(box, score, label), (), ...]
        """
        batch_size = len(pred_boxes_3d)
        num_max_rois = max([len(preds[0]) for preds in pred_boxes_3d])
        num_max_rois = max(1, num_max_rois)
        pred_boxes = pred_boxes_3d[0][0]

        if len(pred_boxes_3d[0]) == 5:
            use_sem_score = True
        else:
            use_sem_score = False

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois,self.num_class))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()
        roi_centerness = pred_boxes.new_zeros((batch_size, num_max_rois))
        #add for assign
        # if len(pred_boxes_3d[0]) == 4:
        #     roi_stage_one_assigned_ids = pred_boxes.new_zeros((batch_size, num_max_rois))
        if use_sem_score:
            roi_sem_scores = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes_3d[0][3].shape[-1]))

        for bs_idx in range(batch_size):
            num_boxes = len(pred_boxes_3d[bs_idx][0])            
            rois[bs_idx, :num_boxes, :] = pred_boxes_3d[bs_idx][0]
            roi_scores[bs_idx, :num_boxes, :] = pred_boxes_3d[bs_idx][1]
            roi_labels[bs_idx, :num_boxes] = pred_boxes_3d[bs_idx][2]
            roi_centerness[bs_idx, :num_boxes] = pred_boxes_3d[bs_idx][3].squeeze(-1)
            #added for assign
            # if len(pred_boxes_3d[0]) == 4:
            #     roi_stage_one_assigned_ids[bs_idx,:num_boxes] = pred_boxes_3d[bs_idx][3]
            if use_sem_score:
                roi_sem_scores[bs_idx, :num_boxes] = pred_boxes_3d[bs_idx][3]
        
        # converse heading to pcdet heading！
        rois[..., 6] *= -1
        if use_sem_score:
            return rois, roi_scores, roi_labels, roi_sem_scores, batch_size
        # else:
        # if len(pred_boxes_3d[0]) == 4:
        #     return rois, roi_scores, roi_labels, batch_size,roi_stage_one_assigned_ids
        else:
            return rois, roi_scores, roi_labels, batch_size,roi_centerness
    
    
    
    def assign_targets(self, input_dict):
        with torch.no_grad():
            targets_dict = self.proposal_target_layer(input_dict)
        batch_size = input_dict['batch_size']
        rois = targets_dict['rois'] 
        gt_of_rois = targets_dict['gt_of_rois'] 
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()
        gt_label_of_rois = targets_dict['gt_label_of_rois'] # b, num_max_rois
        
        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        # also change gt angle to 0 ~ 2pi
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center 
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry 

        if self.code_size > 6:
            # transfer LiDAR coords to local coords
            gt_of_rois = common_utils.rotate_points_along_z(
                points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
            ).view(batch_size, -1, gt_of_rois.shape[-1])

            # flip orientation if rois have opposite orientation
            heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
            heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
            flag = heading_label > np.pi
            heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
            heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

            gt_of_rois[:, :, 6] = heading_label

        targets_dict['gt_of_rois'] = gt_of_rois
        
     
        return targets_dict

    
    def forward_train(self, input_dict):
      
        pred_boxes_3d = input_dict['pred_bbox_list']
        rois = self.reorder_rois_for_refining_v1(pred_boxes_3d)
        #added for assign_targets
        if self.enlarge_ratio:
            rois[..., 3:6] *= self.enlarge_ratio
        input_dict['rois'] = rois

        # assign targets
        targets_dict = self.assign_targets(input_dict)
        input_dict.update(targets_dict)

        # roi pooling
        pooled_features = self.roi_grid_pool(input_dict)  
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)  
        if not self.use_center_pooling:
            shared_features = self.shared_fc_layer(pooled_features) 
        else:
            shared_features = pooled_features
        
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features)) # (BN, 6)
        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features)) # (BN, 6)

        input_dict['rcnn_reg'] = rcnn_reg
        input_dict['rcnn_cls'] = rcnn_cls
        
        loss = self.loss(input_dict)

        return loss
    
    def forward_test(self, input_dict,img_meta):
        pred_boxes_3d = input_dict['pred_bbox_list']
        # preprocess rois, padding to same number
        if len(pred_boxes_3d[0]) == 5:
            use_sem_score = True
            rois, roi_scores, roi_labels, roi_sem_scores, batch_size = self.reoder_rois_for_refining_test(pred_boxes_3d) 
        else:
            rois, roi_scores, roi_labels, batch_size,roi_centerness = self.reoder_rois_for_refining_test(pred_boxes_3d)
            use_sem_score = False
        
        
        input_dict['rois'] = rois
        input_dict['roi_scores'] = roi_scores
        input_dict['roi_labels'] = roi_labels
        input_dict['roi_centerness'] = roi_centerness
        
        if use_sem_score:
            input_dict['roi_sem_scores'] = roi_sem_scores
        input_dict['batch_size'] = batch_size        

        # roi pooling
        pooled_features = self.roi_grid_pool(input_dict)  
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)  
        if not self.use_center_pooling:
            shared_features = self.shared_fc_layer(pooled_features) 
        else:
            shared_features = pooled_features
        
        
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features)) 
        #add cls pred
        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))
        
        input_dict['rcnn_reg'] = rcnn_reg
        input_dict['rcnn_cls'] = rcnn_cls

        batch_size = input_dict['batch_size']
    
        results = self.get_boxes(input_dict, img_meta) 

        return results
    
    
    
    
    
    def get_boxes_cls(self, input_dict, img_meta):
        code_size = self.code_size
        batch_size = input_dict['batch_size']
        #rcnn_cls = None
        rcnn_cls = input_dict['rcnn_cls']
        roi_sem_scores = input_dict.get('roi_sem_scores', None)
        batch_box_preds = input_dict['rois'][...,0:code_size]
        batch_cls_preds = rcnn_cls.view(batch_size,-1,self.num_class)

        results = []
        for bs_id in range(batch_size):
            # nms
            boxes = batch_box_preds[bs_id]
            # scores = roi_scores[bs_id]
            # labels = roi_labels[bs_id]
            # scores = batch_cls_preds[bs_id].squeeze(-1)
            scores = F.softmax(batch_cls_preds[bs_id], dim=-1)[:, :-1]
            labels = torch.argmax(scores, dim=1)#n+1
            #scores = F.softmax(batch_cls_preds[bs_id], dim=-1) #n
            # num_classes = scores.shape[1]
            # labels = torch.arange(
            # num_classes,
            # device=scores.device).unsqueeze(0).repeat(
            #     len(batch_cls_preds[bs_id]), 1).flatten(0, 1)
            # scores, topk_idx = scores.flatten(0, 1).topk(
            # self.test_cfg.topk_insts, sorted=True)
            # labels = labels[topk_idx]

            # topk_idx = torch.div(topk_idx, num_classes, rounding_mode='floor')
            # pred_bboxes = boxes[topk_idx]

            
            #scores = roi_scores[bs_id]
            # labels = roi_labels[bs_id]
            #max_scores, labels = scores.max(dim=1)
            #result = self._single_scene_multiclass_nms(pred_bboxes, scores, labels, img_meta[bs_id])
            #result = self._nms(boxes, scores, labels, img_meta[bs_id])
            result = self._nms_v1(boxes, scores, img_meta[bs_id])
            #result = self.class_agnostic_nms(boxes, scores, img_meta[bs_id])
            results.append(result)
        return results 
    
    def get_boxes(self, input_dict, img_meta):
        batch_size = input_dict['batch_size']
        #rcnn_cls = None
        rcnn_cls = input_dict['rcnn_cls']
        rcnn_reg = input_dict['rcnn_reg']
        roi_labels = input_dict['roi_labels']
        roi_scores = input_dict['roi_scores']
        roi_centerness = input_dict['roi_centerness'].unsqueeze(-1)
        # batch_box_preds = input_dict['rois'][...,0:code_size]
        roi_sem_scores = input_dict.get('roi_sem_scores', None)
        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_size, rois=input_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg, roi_labels= roi_labels, roi_sem_scores=roi_sem_scores
            )
        input_dict['cls_preds_normalized'] = False
        if not input_dict['cls_preds_normalized'] and batch_cls_preds is not None:
            batch_cls_preds = torch.sigmoid(batch_cls_preds)* roi_centerness.sigmoid() #正则化,centerness is important
            #batch_cls_preds = torch.sigmoid(batch_cls_preds)* roi_centerness #already sigmoid
            
            #batch_cls_preds = F.softmax(batch_cls_preds,dim=-1)* roi_centerness.sigmoid()  #change sigmoid or softmax
        input_dict['batch_cls_preds'] = batch_cls_preds # B,N
        input_dict['batch_box_preds'] = batch_box_preds 

        results = []
        for bs_id in range(batch_size):
            # nms
            #boxes = input_dict['rois'][bs_id]
            boxes = batch_box_preds[bs_id] 
            #scores = roi_scores[bs_id]
            # labels = roi_labels[bs_id]
            #scores = batch_cls_preds[bs_id].squeeze(-1)
            # labels = torch.argmax(scores, dim=1)
            # scores = F.softmax(batch_cls_preds[bs_id], dim=-1)[:, :-1] #n+1
            #scores = F.softmax(batch_cls_preds[bs_id], dim=-1) #n
            scores1 = roi_scores[bs_id]
            scores2 = batch_cls_preds[bs_id].squeeze(-1)
            w1 = 0.5
            w2 = 0.5
            scores = w1*scores1+w2*scores2
            # for tranformer
            # num_classes = scores.shape[1]
            # labels = torch.arange(
            # num_classes,
            # device=scores.device).unsqueeze(0).repeat(
            #     len(batch_cls_preds[bs_id]), 1).flatten(0, 1)
            # scores, topk_idx = scores.flatten(0, 1).topk(
            # self.test_cfg.topk_insts, sorted=True)
            # labels = labels[topk_idx]

            # topk_idx = torch.div(topk_idx, num_classes, rounding_mode='floor')
            # pred_bboxes = boxes[topk_idx]

            
            #scores = roi_scores[bs_id]
            # labels = roi_labels[bs_id]
            #max_scores, labels = scores.max(dim=1)
            #result = self._single_scene_multiclass_nms(pred_bboxes, scores, labels, img_meta[bs_id])
            # result = self._nms(boxes, scores, labels, img_meta[bs_id])
            result = self._nms_v1(boxes, scores, img_meta[bs_id])
            #result = self.class_agnostic_nms(boxes, scores, img_meta[bs_id])
            results.append(result)
        return results
    
    def _single_scene_multiclass_nms(self,bboxes,scores,labels,img_meta):
        classes = labels.unique()
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for class_id in classes:
            ids = scores[labels == class_id] > self.test_cfg.nms_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[labels == class_id][ids]
            class_bboxes = bboxes[labels == class_id][ids]
            class_labels = labels[labels == class_id][ids]
            if yaw_flag:
                nms_ids = nms3d(class_bboxes, class_scores, self.test_cfg.nms_cfg.iou_thr)
            else:
                class_bboxes = torch.cat(
                        (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                        dim=1)
                nms_ids = nms3d_normal(class_bboxes, class_scores,self.test_cfg.nms_cfg.iou_thr)

            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(class_labels[nms_ids])

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))
        
        if yaw_flag:
            # converse pcdet heading to original heading!
            nms_bboxes[..., 6] *= -1
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))
        
        return nms_bboxes, nms_scores, nms_labels

    def class_agnostic_nms(self, bboxes, scores, img_meta, sem_scores=None):
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        max_scores, labels = scores.max(dim=1)
        if yaw_flag:
            nms_function = nms3d
        else:
            bboxes = torch.cat((
                    bboxes, torch.zeros_like(bboxes[:, :1])), dim=1)
            nms_function = nms3d_normal
        
        ids = max_scores > self.test_cfg.nms_cfg.score_thr
        
        if not ids.any():
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0,))
            nms_labels = bboxes.new_zeros((0,))
            if sem_scores is not None:
                nms_sem_scores = bboxes.new_zeros((0, n_classes))
        else:
            class_bboxes = bboxes[ids]
            class_scores = max_scores[ids]
            class_labels = labels[ids]
            if sem_scores is not None:
                class_sem_scores = sem_scores[ids] # n, n_class
            # correct_heading
            correct_class_bboxes = class_bboxes.clone()
            # if yaw_flag:
            #     correct_class_bboxes[..., 6] *= -1
            nms_ids = nms_function(correct_class_bboxes, class_scores, self.test_cfg.nms_cfg.iou_thr)
            nms_bboxes = class_bboxes[nms_ids]
            nms_scores = class_scores[nms_ids]
            nms_labels = class_labels[nms_ids]
        
        if yaw_flag:
            # converse pcdet heading to original heading!
            nms_bboxes[..., 6] *= -1
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels
    
    
    def _nms_v1(self,bboxes,scores,img_meta):
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            #ids = scores[:, i] > .2
            if not ids.any():
                continue
        
            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            
            if yaw_flag:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat((
                    class_bboxes, torch.zeros_like(class_bboxes[:, :1])), dim=1)
                nms_function = nms3d_normal
            
            nms_ids = nms_function(class_bboxes, class_scores, self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(bboxes.new_full(class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0,))
            nms_labels = bboxes.new_zeros((0,))
        
        if yaw_flag:
            # converse pcdet heading to original heading!
            nms_bboxes[..., 6] *= -1
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))
        
        return nms_bboxes, nms_scores, nms_labels
    
    
    def _nms(self, bboxes, scores, labels, img_meta):
        n_classes = self.num_class
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            if scores.ndim == 2:
                ids = (scores[:, i] > self.test_cfg.score_thr) & (bboxes.sum() != 0) # reclass
                #ids = (labels == i) & (scores[:, i] > self.test_cfg.nms_cfg.score_thr) & (bboxes.sum() != 0) # no reclass
            else:
                ids = (labels == i) & (scores > self.test_cfg.score_thr) & (bboxes.sum() != 0)
            if not ids.any():
                continue
            class_scores = scores[ids] if scores.ndim == 1 else scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat((
                    class_bboxes, torch.zeros_like(class_bboxes[:, :1])), dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores, self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(bboxes.new_full(class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0,))
            nms_labels = bboxes.new_zeros((0,))

        if yaw_flag:
            # converse pcdet heading to original heading
            nms_bboxes[..., 6] *= -1
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))
            # fake_heading = nms_bboxes.new_zeros(nms_bboxes.shape[0], 1)
            # nms_bboxes = torch.cat([nms_bboxes[:, :6], fake_heading], dim=1)

        return nms_bboxes, nms_scores, nms_labels
    
    
    
    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds, roi_labels=None, gt_bboxes_3d=None, gt_labels_3d=None, roi_sem_scores=None):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.code_size
        # batch_cls_preds = None
        #batch_cls_preds = cls_preds
        batch_cls_preds = cls_preds.view(batch_size,-1,self.num_class)
        if self.encode_angle_by_sincos:
            batch_box_preds = box_preds.view(batch_size, -1, code_size+1) 
        else:
            batch_box_preds = box_preds.view(batch_size, -1, code_size) 
       
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()[..., :code_size]
        local_rois[:, :, 0:3] = 0
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)
        
        #移到全局坐标系
        if self.code_size > 6:
            roi_ry = rois[:, :, 6].view(-1)
            batch_box_preds = common_utils.rotate_points_along_z(
                batch_box_preds.unsqueeze(dim=1), roi_ry
            ).squeeze(dim=1)

        batch_box_preds[:, 0:3] += roi_xyz 
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)

        return batch_cls_preds, batch_box_preds
    
    def loss(self, input_dict):
        
        rcnn_loss_dict = {}
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(input_dict)
        if not self.use_iou_loss:
            rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(input_dict)
            rcnn_loss_dict['rcnn_loss_reg'] = rcnn_loss_reg
        else:
            rcnn_loss_reg, rcnn_loss_iou, reg_tb_dict = self.get_box_reg_layer_loss(input_dict)
            if self.loss_weight.rcnn_reg_weight > 0:
                rcnn_loss_dict['rcnn_loss_reg'] = rcnn_loss_reg
            rcnn_loss_dict['rcnn_loss_iou'] = rcnn_loss_iou
        rcnn_loss_dict['rcnn_loss_cls'] = rcnn_loss_cls
        loss = 0.
        tb_dict = dict()
        for k in rcnn_loss_dict.keys():
            loss += rcnn_loss_dict[k]
            tb_dict[k] = rcnn_loss_dict[k]
        #tb_dict['loss_two_stage'] = loss.item()
        return loss, tb_dict
    
    def get_box_cls_layer_loss(self, forward_ret_dict):
        rcnn_cls = forward_ret_dict['rcnn_cls'] #[2048xnum_classes]
        # rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        rcnn_gt_labels = forward_ret_dict['gt_label_of_rois'].view(-1) #[2048]
        rcnn_gt_labels = rcnn_gt_labels.long()
        batch_loss_cls = self.cls_loss(rcnn_cls, rcnn_gt_labels)
        # # #cls_valid_mask = (rcnn_cls_labels >= 0).float()
        reg_valid_mask  = forward_ret_dict['reg_valid_mask'].view(-1)
        cls_valid_mask =  (reg_valid_mask  > 0) 
        rcnn_loss_cls = torch.sum(batch_loss_cls)/ torch.clamp(cls_valid_mask.sum(), min=1.0)
        
        rcnn_loss_cls = rcnn_loss_cls * self.loss_weight.rcnn_cls_weight
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict
    
    def get_box_reg_layer_loss(self, forward_ret_dict):
        code_size = self.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois'][..., 0:code_size]
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if self.reg_loss_type == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            if code_size > 6:
                rois_anchor[:, 6] = 0

            # encode box
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 6]

            #rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1) #只计算回归前景损失
            rcnn_loss_reg = torch.sum(rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()) / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * self.loss_weight.rcnn_reg_weight
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item
            loss_iou = torch.tensor(0., device=fg_mask.device)
            if self.use_iou_loss and fg_sum > 0:
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()

                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size+1 if self.encode_angle_by_sincos else code_size), batch_anchors
                ).view(-1, code_size)
                #从局部坐标系移到全局
                if self.code_size > 6:
                    roi_ry = batch_anchors[:, :, 6].view(-1)
                    rcnn_boxes3d = common_utils.rotate_points_along_z(
                        rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                    ).squeeze(dim=1)

                rcnn_boxes3d[:, 0:3] += roi_xyz
                loss_iou = self.iou_loss_computer(_bbox_to_loss(rcnn_boxes3d[:, 0:code_size]),
                    _bbox_to_loss(gt_of_rois_src[fg_mask][:, 0:code_size])
                ) #scannet needs trans
                loss_iou = torch.mean(loss_iou) * self.loss_weight.rcnn_iou_weight
                tb_dict['rcnn_loss_iou'] = loss_iou.item()
        else:
            raise NotImplementedError
        
        if not self.use_iou_loss:
            return rcnn_loss_reg, tb_dict
        else:
            return rcnn_loss_reg, loss_iou, tb_dict

    def forward(self, input_dict):
        if self.training:
            return self.forward_train(input_dict)
        else:
            return self.simple_test(input_dict)
    
def _bbox_to_loss(bbox):
    """Transform box to the axis-aligned or rotated iou loss format.

    Args:
        bbox (Tensor): 3D box of shape (N, 6) or (N, 7).

    Returns:
        Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
    """
    # # rotated iou loss accepts (x, y, z, w, h, l, heading)
    if bbox.shape[-1] != 6:
        return bbox

    # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
    return torch.stack(
        (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
            bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
            bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
        dim=-1)


      