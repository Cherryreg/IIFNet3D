# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/detectors/single_stage_sparse.py # noqa
# create by -zyrant

try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

from mmdet3d.core import bbox3d2result, bbox3d_mapping_back
from mmdet3d.models import DETECTORS, build_backbone, build_head, build_neck, build_voxel_encoder
from mmdet3d.models.detectors.base import Base3DDetector
import torch
import numpy as np
from torch_scatter import scatter_mean
from mmcv.ops import nms3d, nms3d_normal
import os
import csv
from mmdet3d.models.fusion_layers.point_fusion import point_sample
from mmdet3d.core.bbox.structures import get_proj_mat_by_coord_type
from torch import nn
from functools import partial

@DETECTORS.register_module()
class IIFNet3DDetector(Base3DDetector):

    r"""Two stage detector based on MinkowskiEngine `GSDN
    <https://arxiv.org/abs/2006.12356>`_.

    Args:
        backbone (dict): Config of the backbone.
        head (dict): Config of the head.
        voxel_size (float): Voxel size in meters.
        neck (dict): Config of the neck.
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
        pretrained (str, optional): Deprecated initialization parameter.
            Defaults to None.
    """
    _version = 2

    def __init__(self,
                img_backbone,
                 img_neck,
                 backbone,
                 voxel_size,
                 neck = None,
                 cpg_encoder= None,
                 cpg_head = None,
                 roi_head = None,
                 train_cfg = None,
                 test_cfg = None,
                 init_cfg= None,
                 pretrained=None) -> None:
        super().__init__(
            init_cfg=init_cfg)
        self.img_backbone = build_backbone(img_backbone)
        self.img_neck = build_neck(img_neck)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)

        cpg_head.update(train_cfg=train_cfg)
        cpg_head.update(test_cfg=test_cfg)
        roi_head.update(train_cfg=train_cfg)
        roi_head.update(test_cfg=test_cfg)
        self.cpg_head = build_head(cpg_head)
        self.roi_head = build_head(roi_head)
        self.voxel_size = voxel_size
        
        self.use_voxel_encoder = False
        if cpg_encoder is not None:
            self.cpg_encoder = build_voxel_encoder(cpg_encoder)
            self.use_voxel_encoder = True
    
  
    def init_weights(self, pretrained=None):
        # self.img_backbone.init_weights()
        # self.img_neck.init_weights()
        for param in self.img_backbone.parameters():
            param.requires_grad = False
        for param in self.img_neck.parameters():
            param.requires_grad = False
        self.img_backbone.eval()
        self.img_neck.eval()
        self.backbone.init_weights()
        self.cpg_head.init_weights()
        self.roi_head.init_weights()
    
    def _f(self, x, img_features, img_metas, img_shape):
        points = x.decomposed_coordinates #坐标
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size
        projected_features = []
        for point, img_feature, img_meta in zip(points, img_features, img_metas):
            coord_type = 'DEPTH'
            img_scale_factor = (
                point.new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = (
                point.new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = get_proj_mat_by_coord_type(img_meta, coord_type)
            projected_features.append(point_sample(
                img_meta=img_meta,
                img_features=img_feature.unsqueeze(0),
                points=point,
                proj_mat=point.new_tensor(proj_mat),
                coord_type=coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img_shape[-2:],
                img_shape=img_shape[-2:],
                aligned=True,
                padding_mode='zeros',
                align_corners=True))

        projected_features = torch.cat(projected_features, dim=0)
        projected_features = ME.SparseTensor(
            projected_features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager)
        
        return projected_features
    

    def extract_feat(self, *args):
        """Just implement @abstractmethod of BaseModule."""

    def extract_feats(self, points, superpoints,img_metas,img):
        """Extract features from points.

        Args:
            points (list[Tensor]): Raw point clouds.

        Returns:
            SparseTensor: Voxelized point clouds.
        """
        with torch.no_grad():
            x = self.img_backbone(img)
            img_features = self.img_neck(x)[0]
        
        coordinates, features = ME.utils.batch_sparse_collate( 
            [(p[:, :3] / self.voxel_size, p[:, 3:]) for p in points],
            device=points[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)    
        
        #unprojected first for roi head
        feat_2d = self._f(x,img_features=img_features, img_metas=img_metas, img_shape=img.shape)
        
        x = self.backbone(x)
        
        decode_res = x
        
        
        if self.with_neck:
            x = self.neck(x)
        if self.use_voxel_encoder:
            x = self.cpg_encoder(x, points, superpoints,img_metas)
        
            
        return x,feat_2d,decode_res #121600x64

    def forward_train(self, points, superpoints, 
                        gt_bboxes_3d, gt_labels_3d,
                        pts_semantic_mask=None,
                        pts_instance_mask=None, img_metas=None,img=None):
        """Forward of training.

        Args:
            points (list[Tensor]): Raw point clouds.
            gt_bboxes (list[BaseInstance3DBoxes]): Ground truth
                bboxes of each sample.
            gt_labels(list[torch.Tensor]): Labels of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            dict: Centerness, bbox and classification loss values.
        """
    
        x,feat_2d,decode_res = self.extract_feats(points, superpoints, img_metas,img) #feat_2d -sparsetensor,
        
        if not self.use_voxel_encoder:
            x = self.head.forward(x, points, superpoints)
        
        
        #pred
        center_preds, bbox_preds, cls_preds,points_pred,vote_offsets,vote_voxel_points, orgin_superpoints = self.cpg_head(x)
        
        rois,bbox_targets, cls_targets = self.cpg_head._get_rois_targets(
         points_pred, bbox_preds, cls_preds, gt_bboxes_3d, gt_labels_3d,img_metas)
      
         
        
        losses = dict()
    
        cpg_losses = self.cpg_head._loss(center_preds, bbox_preds, cls_preds, points_pred, vote_offsets,vote_voxel_points, orgin_superpoints, [points],
                          pts_semantic_mask, pts_instance_mask, gt_bboxes_3d, gt_labels_3d, img_metas)
        
        losses.update(cpg_losses)
        
        decode_out = [None, decode_res, feat_2d, x['decode_out']] #x[decode_out'] is sp feats
        
        stage_two_dict = dict()
        stage_two_dict['middle_feature_list'] = decode_out
       
        
        gt_bboxes = []
        for i in range(len(gt_bboxes_3d)):
            bbox = torch.cat((gt_bboxes_3d[i].gravity_center, gt_bboxes_3d[i].tensor[:, 3:]), dim=1)
            bbox = bbox.to(gt_bboxes_3d[i].device)
            if bbox.shape[1] == 6:
                fake_heading = bbox.new_zeros(bbox.shape[0], 1)
                bbox = torch.cat([bbox[:, :6], fake_heading], dim=1) #all bboxes have 7 
            #!convert heading for criterion
            # bbox[...,6] *= -1 
            gt_bboxes.append(bbox)
        stage_two_dict['gt_bboxes_3d'] = gt_bboxes
        stage_two_dict['gt_labels_3d'] = gt_labels_3d
        stage_two_dict['pred_bbox_list'] = rois
        stage_two_dict['batch_gt_of_rois'] = bbox_targets
        stage_two_dict['batch_gt_label_of_rois'] = cls_targets
        stage_two_dict['batch_size'] = len(img_metas)
        
        loss,roi_loss = self.roi_head.forward_train(stage_two_dict) #criterion
        
        losses.update(roi_loss)
        
        return losses
    
    def forward_test(self, points,superpoints,img_metas, img=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(points, 'points'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(points)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(points), len(img_metas)))

        if num_augs == 1:
            img = [img] if img is None else img
            return self.simple_test(points[0], superpoints[0], img_metas[0], img[0], **kwargs)
        else:
            return self.aug_test(points, superpoints, img_metas, img, **kwargs)

    def simple_test(self, points, superpoints, img_metas, img, *args, **kwargs):
        """Test without augmentations.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
                

        x,feat_2d,decode_res = self.extract_feats(points, superpoints, img_metas, img) #feat_2d -sparsetensor,
        rois = self.cpg_head.forward_test(x,img_metas)
        
        decode_out = [None, decode_res, feat_2d, x['decode_out']] #x[decode_out'] is sp feats
        #decode_out = [None, None, feat_2d, decode_res] 
        
        stage_two_dict = dict()
        stage_two_dict['middle_feature_list'] = decode_out
        
        stage_two_dict['pred_bbox_list'] = rois
        stage_two_dict['batch_size'] = len(img_metas)
        
        bbox_list = self.roi_head.forward_test(stage_two_dict,img_metas)
    
    
        
        
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
    
    
    

    def aug_test(self, points, superpoints, img_metas, img, **kwargs):
        """Test with augmentations.

        Args:
            points (list[list[torch.Tensor]]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        # only support aug_test for one sample
        aug_bboxes = []
        for point, superpoint, img_meta in zip(points, superpoints, img_metas):
            x = self.extract_feats(point, superpoint)
            if not self.use_voxel_encoder:
                x = self.head.forward(x, point, superpoint)
            bbox_list = self.head.forward_test(x, img_meta)

            # bbox_results = [
            #     bbox3d2result(bboxes, scores, labels)
            #     for bboxes, scores, labels in bbox_list
            # ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        recovered_bboxes = []
        recovered_scores = []
        recovered_labels = []

        final_results = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            scale_factor = img_info[0]['pcd_scale_factor']
            pcd_horizontal_flip = img_info[0]['pcd_horizontal_flip']
            pcd_vertical_flip = img_info[0]['pcd_vertical_flip']
            recovered_scores.append(bboxes[1])
            recovered_labels.append(bboxes[2])
            bboxes = bbox3d_mapping_back(bboxes[0], scale_factor,
                                        pcd_horizontal_flip, pcd_vertical_flip)
            recovered_bboxes.append(bboxes)

        aug_bboxes = recovered_bboxes[0].cat(recovered_bboxes)
        aug_scores = torch.cat(recovered_scores, dim=0)
        aug_labels = torch.cat(recovered_labels, dim=0)
        res_bboxes, res_scores, res_labels = self._single_scene_multiclass_nms(
            aug_bboxes, aug_scores, aug_labels, img_metas[0][0])

        final_results.append([res_bboxes, res_scores, res_labels])
        final_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in final_results
            ]

        return final_results
    
    def _single_scene_multiclass_nms(self, aug_bboxes, aug_scores, aug_labels, img_meta):
        """Multi-class nms for a single scene.

        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.

        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """
        
        yaw_flag = aug_bboxes.with_yaw
        bboxes = torch.cat((aug_bboxes.gravity_center, aug_bboxes.tensor[:, 3:]),
                          dim=1)
        nms_bboxes, nms_scores, nms_labels = [], [], []
        class_ids = torch.unique(aug_labels)
        for class_id in class_ids:
            class_inds = (aug_labels == class_id)
            bboxes_i = bboxes[class_inds]
            scores_i = aug_scores[class_inds]
            labels_i = aug_labels[class_inds]

            if yaw_flag:
                nms_function = nms3d
            else:
                nms_function = nms3d_normal

            selected = nms_function(bboxes_i, scores_i,
                                   self.head.test_cfg.iou_thr)
            
            nms_bboxes.append(bboxes_i[selected, :])
            nms_scores.append(scores_i[selected])
            nms_labels.append(labels_i[selected])

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))


        if yaw_flag:
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
    
    
    #剪枝超点
    def sp_pruning(x,gt_bboxes_3d):
        pass

   