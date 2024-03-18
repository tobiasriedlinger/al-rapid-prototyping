_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    # '../_base_/datasets/mnist_detection_active_mdcomp.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),

    ])
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=5, norm_type=2))
# dataset settings
dataset_id = "mnist_det_active"
dataset_type = 'ALDetection'
data_root = '/home/USR/active_learning_od/nutshell_mnist/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),

    dict(type='Resize', img_scale=(300, 300), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
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
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
classes = [str(x) for x in range(10)]
data = dict(

    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train/annotations/mnistdetection20k_train_instances.json',
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
# evaluation = dict(interval=1, metric='bbox')


model = dict(
    type='RetinaNet',
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
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
        metrics=False))

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]

# runner = dict(type='EpochBasedRunner', max_epochs=200)
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=10000)

evaluation = dict(interval=500, metric='bbox')
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8000, 9000])

checkpoint_config = dict(interval=runner['max_iters'])
workflow = [('train', 1) , ('val', 1)]

# Active Learning Settings
splits = dict(
    initial=dict(
        size=50,
        ratio=None
    ),
    validation=dict(
        size=500
    ),
    train_track=dict(
        size=200
    ),
    query=dict(
        pool_size=2000,
        method="entropy",
        distance="cosine",
        aggregation="average",
        weights="class_wise",
        size=25,
        ratio=None,
        score_thr=0.5
    )
)

al_run = dict(
    run_count=0,
    initial_step=0,
    total_num_counts=5
)

metrics = False

# load_from = "/home/USR/mm-detection-active-learning-for-object-detection/checkpoints/retinanet/mnist_det_active/random/run_0/step_0/latest.pth"

output_dir = "/home/USR/mm-detection-active-learning-for-object-detection/results/al_runs/retinanet"