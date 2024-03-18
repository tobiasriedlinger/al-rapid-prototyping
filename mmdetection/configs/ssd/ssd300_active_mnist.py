_base_ = ['../_base_/datasets/mnist_detection.py', '../_base_/default_runtime.py', '../_base_/schedules/schedule_2x.py'
          ]

input_size = 300
model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='SSDNeck',
        in_channels=(512, 1024),
        out_channels=(512, 1024, 512, 256, 256, 256),
        level_strides=(2, 2, 1, 1),
        level_paddings=(1, 1, 0, 0),
        l2_norm_scale=20),
    bbox_head=dict(
        type='SSDHead',
        in_channels=(512, 1024, 512, 256, 256, 256),
        num_classes=10,
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=input_size,
            basesize_ratio_range=(0.15, 0.9),
            strides=[8, 16, 32, 64, 100, 300],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2])),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200))
cudnn_benchmark = True

# dataset settings
dataset_id = "mnist_det"
dataset_type = 'ALDetection'
data_root = '/home/USR/active_learning_od/nutshell_mnist/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
classes = [str(x) for x in range(10)]
data = dict(
    
    samples_per_gpu=6,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train/annotations/mnistdetection_train_instances.json',
        img_prefix=data_root + 'train/img/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,

        ann_file=data_root + 'val/annotations/mnistdetection_val_instances.json',
        img_prefix=data_root + 'val/img/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,

        ann_file=data_root + 'test/annotations/mnistdetection_test_instances.json',
        img_prefix=data_root + 'test/img/',
        pipeline=test_pipeline))
evaluation = dict(interval=5, metric='bbox')
# optimizer
optimizer = dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(_delete_=True)
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]

runner = dict(type='EpochBasedRunner', max_epochs=150)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[110, 140])

checkpoint_config = dict(interval=runner["max_epochs"])
workflow = [('train', 1), ('val', 1)]

# Active Learning Settings
splits = dict(
    initial=dict(
        size=100,
        ratio=None
    ),
    validation=dict(
        size=500
    ),
    train_track=dict(
        size=100
    ),
    query=dict(
        pool_size=2000,
        method="random",
        # insert aggregation method
        size=100,
        ratio=None,
        score_thr=0.1
    )
)

al_run = dict(
    run_count=0,
    initial_step=0,
    total_num_counts=55
)

metrics = False

output_dir = "/home/USR/active_learning_od/mmdetection/checkpoints/ssd"
