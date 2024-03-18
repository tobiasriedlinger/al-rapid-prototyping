_base_ = ['../_base_/datasets/bdd.py',
        #   '../_base_/schedules/schedule_1x.py',
          '../_base_/default_runtime.py']

model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
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

log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
runner = dict(type='IterBasedRunner', max_iters=175000)
checkpoint_config = dict(interval=runner["max_iters"])
evaluation = dict(interval=10000, metric=['bbox'])

dataset_id = 'bdd'
dataset_type = 'ALDetection'
data_root = '/home/USR/dataset_ground_truth/bdd100k/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(type='Expand', mean=[0, 0, 0], to_rgb=True, ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

classes = ("pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "trafficlight", "trafficsign")
data = dict(
    
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'bdd100k_train_annotations_clear_daytime.json',
        img_prefix='/home/dataset_sync/datasets/BDD100k/images/100k/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,

        ann_file=data_root + 'bdd100k_val_annotations_clear_daytime.json',
        img_prefix='/home/dataset_sync/datasets/BDD100k/images/100k/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,

        ann_file=data_root + 'bdd100k_test_annotations_clear_daytime.json',
        img_prefix='/home/dataset_sync/datasets/BDD100k/images/100k/val',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# optimizer
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.1,
    step=[150000, 170000])
splits = dict(
    initial=dict(size=1000, ratio=None),
    validation=dict(size=500),
    train_track=dict(size=500),
    query=dict(
        pool_size=2000,
        method='random',
        n_samples=10,
        distance='l2',
        aggregation='sum',
        weights='class_wise',
        size=500,
        ratio=None,
        score_thr=0.5))
al_run = dict(run_count=0, initial_step=0, total_num_counts=8)
metrics = False

output_dir = "/home/USR/results_aggr/al_runs/retinanet"