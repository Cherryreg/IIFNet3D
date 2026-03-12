plugin = True
plugin_dir = "projects/IIFNet/"
# plugin_dir = "projects.IIFNet"
voxel_size = .02

model = dict(
    type='IIFNet3DDetector_fast',
    voxel_size=voxel_size,
    backbone=dict(
        type='BiResNet',
        in_channels = 3,
        out_channels = 64),
    cpg_encoder=dict(
        type='CPG_encoder', 
        in_channels = 64,
        local_k = 8,
        voxel_size = voxel_size,
        latter_voxel_size = voxel_size * 2,
        feat_channels = (64, 128, 128),
        with_xyz = True,
        with_distance = False, # No used
        with_cluster_center = False, # No used
        with_superpoint_center = True,
        mode='max'),
    cpg_head=dict(
        type='CPGHead',
        pred_layer_cfg=dict(
            in_channels = 390, 
            cls_linear_channels=(256, 256),
            reg_linear_channels=(256, 256),
            center_linear_channels=(256, 256)),
        norm_cfg=dict(type='LN', eps=1e-3),
        n_reg_outs = 6,
        n_classes = 18,
        roi_fg_ratio = 0.9,
        roi_per_image = 128,
        code_size = 6,
        center_type = 'pow',
        pts_threshold = 18,
        center_loss=dict(type='CrossEntropyLoss', use_sigmoid=True),
        bbox_loss=dict(type='AxisAlignedIoULoss', mode='diou'),
        cls_loss=dict(type='FocalLoss'),
        vote_loss= dict(type='SmoothL1Loss', beta = 0.04, reduction='sum'),
        cls_cost = dict(type='FocalLossCost', weight=1),
        reg_cost = dict(type='IoU3DCost', iou_mode='diou', weight=1),
        ),
    roi_head=dict(
        type='IIFROIHead',
        n_classes=18,
        grid_size=7,
        middle_feature_source=[1,2,3],#rgb+3dbackbone+superpoint feats
        voxel_size=voxel_size,
        coord_key=2,
        mlps = [[64,128,128],[256,128,128],[390,128,128]],
        code_size=6,
        encode_sincos=False,
        roi_per_image=128, #no use
        roi_fg_ratio=0.9, #no use
        reg_fg_thresh=0.3, 
        roi_conv_kernel=5,
        enlarge_ratio=False,
        use_iou_loss=True,
        use_grid_offset=False,
        use_simple_pooling=True,
        use_center_pooling=True,
        loss_weights=dict(
            rcnn_cls_weight=1.0,
            rcnn_reg_weight=0.5,
            rcnn_iou_weight=1.0,
            code_weight=[1., 1., 1., 1., 1., 1.]
        ),
        iou_loss=dict(type='AxisAlignedIoULoss', mode='diou', reduction='none'),
    ),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=0.5, score_thr=0))


# dataset settings
dataset_type = 'SPScanNetDataset'
data_root = './data/scannet/'
class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadSuperPointsFromFile'), 
    dict(
        type = 'LoadFusedPointsFromFile',  #added for fuse points
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_mask_3d=True,
        with_seg_3d=True),
   
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='PointSegClassMapping',
        valid_cat_ids=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34,
                       36, 39)),
    # We do not sample 100k points for ScanNet, as very few scenes have
    # significantly more then 100k points. So we sample 33 to 100% of them.
    # dict(type='SPPointSample', num_points=n_points),
    dict(type='SPPointSample', num_points=0.33),  #modify for ff
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-.02, 0.02],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='SPDefaultFormatBundle3D', class_names=class_names), #modify for ff
    dict(
        type='Collect3D', #added for fuse points Collect3D_scannet_ff
        keys=['points','superpoints', 'fused_points', 'gt_bboxes_3d', 'gt_labels_3d', 'pts_semantic_mask',
            'pts_instance_mask'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadSuperPointsFromFile'),
    dict(
        type = 'LoadFusedPointsFromFile',  #added for fuse points(sp_scannet,loading,transforms_3d,Bundle)
    ),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        # pcd_horizontal_flip = True,
        # pcd_vertical_flip = True,
        transforms=[
            # dict(
            #     type='GlobalRotScaleTrans',
            #     rot_range=[0, 0],
            #     scale_ratio_range=[1., 1.],
            #     translation_std=[0, 0, 0]),
            # dict(
            #     type='RandomFlip3D',
            #     sync_2d=False,
            #     flip_ratio_bev_horizontal=0.5,
            #     flip_ratio_bev_vertical=0.5),
            dict(type='NormalizePointsColor', color_mean=None),
            dict(
                type='SPDefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'superpoints', 'fused_points']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=15,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'scannet_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=True,
            classes=class_names,
            box_type_3d='Depth',
            use_2d=True)), #if use 2d load fused points
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth',
        use_2d=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth',
        use_2d=True))


optimizer = dict(type='AdamW', lr=.001, weight_decay=.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[9, 12,15])
runner = dict(type='EpochBasedRunner', max_epochs=20)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

checkpoint_config = dict(interval=1, max_keep_ckpts=10)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

