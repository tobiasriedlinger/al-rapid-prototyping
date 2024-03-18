_base_ = './yolox_s_8x8_300e_coco.py'

data_root = '/home/USR/active_learning_od/nutshell_mnist/'

# model settings
model = dict(
    backbone=dict(deepen_factor=0.33, widen_factor=0.375),
    neck=dict(in_channels=[96, 192, 384], out_channels=96),
    bbox_head=dict(in_channels=96, feat_channels=96),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65), metrics=False))

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_scale = (320, 320)

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=114.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=(320, 320), pad_val=114.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

classes = [str(x) for x in range(10)]
dataset_type = 'ALDetection'

data = dict(
    samples_per_gpu=32,
    # samples_per_gpu=32,
    workers_per_gpu=5,
    train=dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'train/annotations/mnistdetection20k_train_instances.json',
        img_prefix=data_root + 'train/img/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline,
    dynamic_scale=img_scale),
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

resume_from = None
interval = 5

# Execute in the order of insertion when the priority is the same.
# The smaller the value, the higher the priority
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
    dict(
        type='SyncRandomSizeHook',
        ratio_range=(10, 20),
        img_scale=img_scale,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=15,
        interval=interval,
        priority=48),
    dict(type='ExpMomentumEMAHook', resume_from=resume_from, priority=49)
]
checkpoint_config = dict(interval=interval)
evaluation = dict(interval=interval, metric='bbox')

lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=15,
    min_lr_ratio=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=70)

# lr_config = dict(
#     _delete_=True,
#     policy='YOLOX',
#     warmup='linear',
#     by_epoch=False,
#     warmup_by_epoch=False,
#     warmup_ratio=1,
#     warmup_iters=1000,  # 5 epoch
#     # num_last_epochs=15,
#     min_lr_ratio=0.0005)
# dict(_delete_=True, type='IterBasedRunner', max_iters=7500)