plugin = True
plugin_dir = "projects/IIFNet/"
voxel_size = .02

model = dict(
    type='IIFNet3DDetector',
    voxel_size=voxel_size,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    backbone=dict(
        type='BiResNet',
        in_channels = 3,
        out_channels = 64),
    cpg_encoder=dict(
        type='CPG_encoder', 
        in_channels=64,
        local_k = 8,
        voxel_size = voxel_size,
        latter_voxel_size = voxel_size * 2,
        feat_channels = (64, 128, 128),
        with_xyz = True,
        with_distance=False, # No used
        with_cluster_center=False, # No used
        with_superpoint_center = True,
        mode='max'),
    cpg_head=dict(
        type='CPGHead',
        pred_layer_cfg=dict(
            in_channels = 390, 
            cls_linear_channels=(256, 256),
            reg_linear_channels=(256, 256),
            center_linear_channels=(256, 256)
            ),
        norm_cfg=dict(type='LN', eps=1e-3),
        n_reg_outs = 8,
        n_classes = 10,
        with_yaw = True,
        roi_fg_ratio = 0.9,
        roi_per_image = 128,
        code_size = 7,
        center_type = 'pow',
        pts_threshold = 18,
        center_loss=dict(type='CrossEntropyLoss', use_sigmoid=True),
        bbox_loss=dict(type='RotatedIoU3DLoss', mode='diou'),
        cls_loss=dict(type='FocalLoss'),
        vote_loss= dict(type='SmoothL1Loss', beta=0.04, reduction='sum', loss_weight=1),
        cls_cost=dict(type='FocalLossCost', weight=1),
        reg_cost=dict(type='IoU3DCost', iou_mode='diou', weight=1),
        # reg_cost=dict(type='BBox3DL1Cost', weight=1) 
        ),
    roi_head=dict(
        type='IIFROIHead',
        n_classes=10,
        grid_size=7,
        middle_feature_source=[1,2,3],#rgb+3d
        voxel_size=voxel_size,
        coord_key=2,
        mlps = [[64,128,128],[256,128,128],[390,128,128]], #add additional backbone feats
        #mlps = [[256,128,128],[64,128,128]], #2d+backbone
        code_size=7,
        encode_sincos=True,
        roi_per_image=128,
        roi_fg_ratio=0.9,
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
            code_weight=[[1., 1., 1., 1., 1., 1., 1., 1.]]
        ),
        iou_loss=dict(type='RotatedIoU3DLoss', mode='diou', reduction='none'),
    ),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=0.5, score_thr=0))


# dataset settings
dataset_type = 'SPSUNRGBDDataset'
data_root = './data/sunrgbd/'
class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')
n_points = 100000
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadSuperPointsFromFile'), 
    dict(type='LoadImageFromFile'), #rgb
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 504), (1333, 528), (1333, 552),
                   (1333, 576), (1333, 600)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='SPPointSample', num_points=n_points),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-.523599, .523599],
        scale_ratio_range=[.85, 1.15],
        translation_std=[.1, .1, .1],
        shift_height=False),
    # dict(type='NormalizePointsColor', color_mean=None),
    dict(type='SPDefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['points', 'superpoints', 'gt_bboxes_3d', 'gt_labels_3d','img'])
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
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        # pcd_horizontal_flip = True,
        transforms=[
            dict(type='Resize', multiscale_mode='value', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            # dict(
            #     type='GlobalRotScaleTrans',
            #     rot_range=[0, 0],
            #     scale_ratio_range=[1., 1.],
            #     translation_std=[0, 0, 0]),
            # dict(
            #     type='RandomFlip3D',
            #     sync_2d=False,
            #     flip_ratio_bev_horizontal=0.5),
            dict(type='SPPointSample', num_points=n_points),
            # dict(type='NormalizePointsColor', color_mean=None),
            dict(
                type='SPDefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'superpoints','img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            modality=dict(use_camera=True, use_lidar=True),
            data_root=data_root,
            ann_file=data_root + 'sunrgbd_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=True,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        modality=dict(use_camera=True, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        modality=dict(use_camera=True, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))



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
# load_from = None
load_from = 'work_dirs/checkpoints/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210323_173222-cad62aeb.pth'  # noqa
resume_from = None
workflow = [('train', 1)]

