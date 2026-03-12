from turtle import forward
import torch
import numpy as np
import torch.nn as nn
#from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu # NOTE:debug!!!!!!!!!!!!!!!!!
# from mmdet3d.ops.pcdet_nms.pcdet_nms_utils import boxes_iou3d_gpu
from mmcv.ops import boxes_iou3d
class ProposalTargetLayer(nn.Module):
    def __init__(self,
                 roi_per_image=128,
                 fg_ratio=0.5,
                 reg_fg_thresh=0.3,
                 cls_fg_thresh=0.55,
                 cls_bg_thresh=0.15,
                 cls_bg_thresh_l0=0.1, #ori-0.1
                 hard_bg_ratio=0.8,
                 n_classes = 11
                 ):
        super(ProposalTargetLayer,self).__init__()
        self.roi_per_image = roi_per_image
        self.fg_ratio = fg_ratio
        self.reg_fg_thresh = reg_fg_thresh
        self.cls_fg_thresh = cls_fg_thresh
        self.cls_bg_thresh = cls_bg_thresh
        self.cls_bg_thresh_l0 = cls_bg_thresh_l0
        self.hard_bg_ratio = hard_bg_ratio
        
        #added_new
        self.n_classes = n_classes
    
    
    #根据一阶段点到框分配
    def forward(self, batch_dict):
        batch_gt_of_rois, batch_gt_label_of_rois,reg_valid_mask = self.sample_rois_for_rcnn_v2(batch_dict=batch_dict)
        batch_rois = batch_dict['rois'].clone() #TODO attention grad
        targets_dict = {'rois': batch_rois,'gt_of_rois': batch_gt_of_rois, 'gt_label_of_rois': batch_gt_label_of_rois,'reg_valid_mask': reg_valid_mask}
        
        return targets_dict
    
    def sample_rois_for_rcnn_v2(self, batch_dict):
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        gt_boxes = batch_dict['gt_bboxes_3d']
        gt_labels = batch_dict['gt_labels_3d']
        roi_per_image = len(rois[0])
        
        #assigned gt in first stage
        ori_gt_of_rois = batch_dict['batch_gt_of_rois']
        ori_gt_label_of_rois = batch_dict['batch_gt_label_of_rois']
        
        gt_code_size = gt_boxes[0].shape[-1]
        
        batch_gt_of_rois = rois.new_zeros(batch_size, roi_per_image, gt_code_size) 
        batch_gt_label_of_rois = rois.new_full((batch_size, roi_per_image), self.n_classes, dtype=torch.long)
        
        reg_mask = rois.new_zeros(batch_size, roi_per_image)
        
        for index in range(batch_size):
            cur_roi,cur_gt,cur_assign_gt,cur_assign_gt_label = rois[index], gt_boxes[index].clone(),ori_gt_of_rois[index].clone(),ori_gt_label_of_rois[index] 
            cur_labels = gt_labels[index]
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt
            
            cur_gt[..., 6] *= -1
            cur_assign_gt[...,6] *= -1
            
            cur_gt = cur_gt.to(cur_roi.device)
            cur_labels = cur_labels.to(cur_gt.device)
            
            stage_one_pos_inds = (cur_assign_gt_label>=0).nonzero().view(-1) 
            
            batch_gt_label_of_rois[index][stage_one_pos_inds] = cur_assign_gt_label[stage_one_pos_inds]
            batch_gt_of_rois[index][stage_one_pos_inds] = cur_assign_gt[stage_one_pos_inds]
            reg_mask[index][stage_one_pos_inds] = 1 #float32
            
             
            #第一阶段负的计算iou
            stage_one_neg_inds = (cur_assign_gt_label < 0).nonzero().view(-1)
            cur_neg_rois = cur_roi[stage_one_neg_inds]
            max_overlaps,gt_assignment = self.get_max_iou_v1(
                        rois=cur_neg_rois, gt_boxes=cur_gt[:, 0:7])
            
              #二阶段判断为正样本的阈值
            fg_thresh = self.reg_fg_thresh
            fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)
            
            if fg_inds.numel() > 0:
                stage_two_pos_inds = stage_one_neg_inds[fg_inds]
                
                batch_gt_of_rois[index][stage_two_pos_inds] = cur_gt[gt_assignment[fg_inds]]
                batch_gt_label_of_rois[index][stage_two_pos_inds] = cur_labels[gt_assignment[fg_inds]]
                reg_mask[index][stage_two_pos_inds] = 1
                
        return batch_gt_of_rois,batch_gt_label_of_rois,reg_mask
    
    
    
    def sample_rois_for_rcnn_v1(self, batch_dict):
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois'] #一阶段预测的数量，分批次
        rois_assigned_ids = batch_dict['rois_assigned_ids']
        gt_boxes = batch_dict['gt_bboxes_3d']
        gt_labels = batch_dict['gt_labels_3d']
        roi_per_image = len(rois[0]) #每批次roi数量一样
        
        # code_size = rois.shape[-1] # 7
        gt_code_size = gt_boxes[0].shape[-1] # 7
       
    
        
        batch_gt_of_rois = rois.new_zeros(batch_size, roi_per_image, gt_code_size) 
        batch_gt_label_of_rois = rois.new_full((batch_size, roi_per_image), self.n_classes, dtype=torch.long)
        
        #回归掩码
        reg_mask = rois.new_zeros(batch_size, roi_per_image)
        
  
   
        
        for index in range(batch_size):
            cur_roi,cur_gt,cur_assign = rois[index], gt_boxes[index].clone(), rois_assigned_ids[index].long()
            cur_labels = gt_labels[index]
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt
            cur_gt[..., 6] *= -1
            #移到设备上
            cur_gt = cur_gt.to(cur_roi.device)
            cur_labels = cur_labels.to(cur_gt.device)
            
            #第一阶段为正的还分给第一阶段预测的gt
            stage_one_pos_inds = (cur_assign >=0).nonzero().view(-1) 
           
            batch_gt_label_of_rois[index][stage_one_pos_inds] = cur_labels[cur_assign[stage_one_pos_inds]]
            batch_gt_of_rois[index][stage_one_pos_inds] = cur_gt[cur_assign[stage_one_pos_inds]]
            reg_mask[index][stage_one_pos_inds] = 1 #float32
            
            #第一阶段负的计算iou
            stage_one_neg_inds = (cur_assign < 0).nonzero().view(-1)
            cur_neg_rois = cur_roi[stage_one_neg_inds]
            max_overlaps,gt_assignment = self.get_max_iou_v1(
                        rois=cur_neg_rois, gt_boxes=cur_gt[:, 0:7])
            #二阶段判断为正样本的阈值
            fg_thresh = self.reg_fg_thresh
            fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)
            
            if fg_inds.numel() > 0:
                stage_two_pos_inds = stage_one_neg_inds[fg_inds]
                
                batch_gt_of_rois[index][stage_two_pos_inds] = cur_gt[gt_assignment[fg_inds]]
                batch_gt_label_of_rois[index][stage_two_pos_inds] = cur_labels[gt_assignment[fg_inds]]
                reg_mask[index][stage_two_pos_inds] = 1
            # else:
            #     stage_two_pos_inds = torch.tensor([], dtype=torch.long, device=assigned_ids.device)
        
        return batch_gt_of_rois,batch_gt_label_of_rois,reg_mask
    
    def sample_rois_for_rcnn(self, batch_dict):
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois'] #一阶段预测的数量
        roi_scores = batch_dict['roi_scores']
        roi_labels = batch_dict['roi_labels']
        gt_boxes = batch_dict['gt_bboxes_3d']
        gt_labels = batch_dict['gt_labels_3d']

        code_size = rois.shape[-1] # 7
        gt_code_size = gt_boxes[0].shape[-1] # 7
        batch_rois = rois.new_zeros(batch_size, self.roi_per_image, code_size) #强制128个
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_per_image, gt_code_size) 
        batch_gt_label_of_rois = rois.new_zeros(batch_size, self.roi_per_image) 
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_per_image)
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_per_image)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_per_image), dtype=torch.long)

        detail_debug = False
        for index in range(batch_size):
            # sun/org
            # cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = \
            #     rois[index], gt_boxes[index].clone(), roi_labels[index], roi_scores[index]
            # NOTE: debug
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = \
                rois[index], gt_boxes[index].clone(), roi_labels[index], roi_scores[index]
            # valid_ind = cur_roi.sum(1) != 0
            # valid_num = max(valid_ind.sum(), 1)
            # cur_roi = cur_roi[0:valid_num]
            # cur_roi_labels = cur_roi_labels[0:valid_num]
            # cur_roi_scores = cur_roi_scores[0:valid_num] # NOTE(lihe): only compute valid roi bboxes
            #
            cur_labels = gt_labels[index]
            # cur_gt = torch.cat((cur_gt.gravity_center.clone(), cur_gt.tensor[:, 3:].clone()), dim=1).to(cur_roi.device) # NOTE
            # converse mmdet3d heading to normal heading !!!!!
            cur_gt[..., 6] *= -1
            
           
            # TODO: check if there are all zeros gt_boxes
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt
            #全空的的label为10
            cur_labels = torch.full((1,), self.n_classes, dtype=torch.int) if len(cur_labels) == 0 else cur_labels
            #cur_labels = torch.zeros((0,), dtype=torch.int) if len(cur_labels) == 0 else cur_labels
            #移到设备上
            cur_gt = cur_gt.to(cur_roi.device)
            cur_labels = cur_labels.to(cur_gt.device)
            # sample roi by each class
            # if cur_labels.numel() != 0:
                # print("gt_labels is empty. Skipping loop.")
            max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                        rois=cur_roi, roi_labels=cur_roi_labels,
                        gt_boxes=cur_gt[:, 0:7], gt_labels=cur_labels.long()
                    )
            # max_overlaps,gt_assignment = self.get_max_iou(
            #             rois=cur_roi, roi_labels=cur_roi_labels,
            #             gt_boxes=cur_gt[:, 0:7], gt_labels=cur_labels.long()
            #         )
            # pos_mask = max_overlaps > min(self.reg_fg_thresh, self.cls_fg_thresh) 
            # cls_targets = torch.where(pos_mask, cur_labels[gt_assignment], self.n_classes)
        
            if detail_debug:
                print("====max_overlaps===: ", max_overlaps.sum())
                seed = np.random.get_state()[1][0]
                print("====cur_seed===: ", seed)
            
            sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)
            # if True:
            #     save_dir = '/data/users/dinglihe01/workspace/CAGroup3D/debug_data/scan_debug/'
            #     sampled_inds = torch.from_numpy(np.load(save_dir + f'sampled_inds_{index}.npy')).cuda()
            # print("====sampled_inds===: ", sampled_inds.sum())
            #从原始采样128个，多[0,0,0,0,0]的默认gt为该场景label=0的gt，否则就给该场景第1个gt。无对应gt则分配给第一个，该场景无gt则分配给全0，标签随便
            batch_rois[index] = cur_roi[sampled_inds]
            batch_roi_labels[index] = cur_roi_labels[sampled_inds]
            batch_roi_ious[index] = max_overlaps[sampled_inds]
            batch_roi_scores[index] = cur_roi_scores[sampled_inds]
            batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]
            batch_gt_label_of_rois[index] = cur_labels[gt_assignment[sampled_inds]]
            # batch_gt_label_of_rois[index] = cls_targets[sampled_inds]
        # TODO: check targets, visualize
        return batch_rois, batch_gt_of_rois, batch_gt_label_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels
        # return batch_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels
    
    def subsample_rois(self, max_overlaps):
        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = int(np.round(self.fg_ratio * self.roi_per_image))
        fg_thresh = self.reg_fg_thresh

        fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)
        easy_bg_inds = ((max_overlaps < self.cls_bg_thresh_l0)).nonzero().view(-1) #1076
        hard_bg_inds = ((max_overlaps < self.reg_fg_thresh) &
                (max_overlaps >= self.cls_bg_thresh_l0)).nonzero().view(-1) #19

        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = self.roi_per_image - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.hard_bg_ratio
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(self.roi_per_image) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = fg_inds[fg_inds < 0] # yield empty tensor

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = self.roi_per_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.hard_bg_ratio
            )
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return sampled_inds

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7)
            roi_labels: (N)
            gt_boxes: (N, )
            gt_labels:

        Returns:

        """
        """
        :param rois: (N, 7)
        :param roi_labels: (N)
        :param gt_boxes: (N, 8)
        :return:
        """
        max_overlaps = rois.new_zeros(rois.shape[0])
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0]) #没有跟gt重合的默认分配给0号gt

        if gt_labels.numel() != 0:
            for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
                roi_mask = (roi_labels == k)
                gt_mask = (gt_labels == k)
                if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                    cur_roi = rois[roi_mask]
                    cur_gt = gt_boxes[gt_mask]
                    original_gt_assignment = gt_mask.nonzero().view(-1)#提取非0元素索引

                    iou3d = boxes_iou3d(cur_roi, cur_gt)  # (M, N)
                    cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
                    max_overlaps[roi_mask] = cur_max_overlaps
                    gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

        return max_overlaps, gt_assignment
    
    @staticmethod
    def get_max_iou(rois,roi_labels, gt_boxes, gt_labels):
        # max_overlaps = rois.new_zeros(rois.shape[0])
        # gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])
        
        #userful for debug
        max_overlaps = rois.new_full((rois.shape[0],), -1, dtype=torch.float)
        gt_assignment = roi_labels.new_full((roi_labels.shape[0],), -1, dtype=torch.long)

        
        iou3d = boxes_iou3d(rois, gt_boxes) # [N,M]
        cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
        
        #  # 先分配给IoU最高的GT
        max_overlaps[:] = cur_max_overlaps
        gt_assignment[:] = cur_gt_assignment
        
        # 对于IoU为0的ROI，先分配给最近的GT
        zero_mask = cur_max_overlaps == 0
        if zero_mask.any():
            # cur_roi = rois[zero_mask]
            roi_centers = rois[:, :3]  # 取ROI的 (x, y, z) 中心点
            gt_centers = gt_boxes[:, :3]  # 取GT的 (x, y, z) 中心点
            
            # 计算ROI与GT的欧几里得距离
            dists = torch.cdist(roi_centers, gt_centers, p=2)  # [L, M]
            min_dists, min_dist_assignment = torch.min(dists, dim=1)
            
            # 只更新那些IoU为0的ROI
            gt_assignment[zero_mask] = min_dist_assignment[zero_mask]
        
        return max_overlaps,gt_assignment
    
    @staticmethod
    def get_max_iou_v1(rois,gt_boxes):
        # max_overlaps = rois.new_zeros(rois.shape[0])
        # gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])
        
        #userful for debug
        max_overlaps = rois.new_full((rois.shape[0],), -1, dtype=torch.float)
        gt_assignment = rois.new_full((rois.shape[0],), -1, dtype=torch.long)

        
        iou3d = boxes_iou3d(rois, gt_boxes) # [N,M]
        cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
        
        #  # 先分配给IoU最高的GT
        max_overlaps[:] = cur_max_overlaps
        gt_assignment[:] = cur_gt_assignment
        
        # 对于IoU为0的ROI，先分配给最近的GT
        zero_mask = cur_max_overlaps == 0
        if zero_mask.any():
            # cur_roi = rois[zero_mask]
            roi_centers = rois[:, :3]  # 取ROI的 (x, y, z) 中心点
            gt_centers = gt_boxes[:, :3]  # 取GT的 (x, y, z) 中心点
            
            # 计算ROI与GT的欧几里得距离
            dists = torch.cdist(roi_centers, gt_centers, p=2)  # [L, M]
            min_dists, min_dist_assignment = torch.min(dists, dim=1)
            
            # 只更新那些IoU为0的ROI
            gt_assignment[zero_mask] = min_dist_assignment[zero_mask]
        
        return max_overlaps,gt_assignment
        
        