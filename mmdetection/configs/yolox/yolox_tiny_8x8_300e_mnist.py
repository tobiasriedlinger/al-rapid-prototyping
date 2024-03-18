<<<<<<< HEAD
_base_ = ['./yolox_s_8x8_300e_coco.py',]
=======
_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']
        #'./yolox_s_8x8_300e_coco.py',]
>>>>>>> d1f949db84ce24686ad482744a0489c3878c7d53
          #'../_base_/datasets/mnist_detection_active_mdcomp.py']

# model settings
model = dict(
<<<<<<< HEAD
    backbone=dict(deepen_factor=0.33, widen_factor=0.375),
    neck=dict(in_channels=[96, 192, 384], out_channels=96),
    bbox_head=dict(num_classes=10, in_channels=96, feat_channels=96),
    init_cfg=dict(type="Pretrained", checkpoint="/home/USR/mm-detection-active-learning-for-object-detection/checkpoints/yolox/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"))
=======
    type='YOLOX',
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.375),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead', num_classes=10, in_channels=96, feat_channels=96),
    init_cfg=dict(type="Pretrained", checkpoint="/home/USR/mm-detection-active-learning-for-object-detection/checkpoints/yolox/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"),
    # train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
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
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65), metrics=False))
# model settings
# model = dict(
#     backbone=dict(deepen_factor=0.33, widen_factor=0.375),
#     neck=dict(in_channels=[96, 192, 384], out_channels=96),
#     bbox_head=dict(num_classes=10, in_channels=96, feat_channels=96),
#     init_cfg=dict(type="Pretrained", checkpoint="/home/USR/mm-detection-active-learning-for-object-detection/checkpoints/yolox/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"))
>>>>>>> d1f949db84ce24686ad482744a0489c3878c7d53

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

<<<<<<< HEAD
img_scale = (300, 300)

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
=======
img_scale = (320, 320)

# train_pipeline = [
#     #dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
#     dict(
#         type='RandomAffine',
#         scaling_ratio_range=(0.5, 1.5),
#         border=(-img_scale[0] // 2, -img_scale[1] // 2)),
#     dict(
#         type='MixUp',
#         img_scale=img_scale,
#         ratio_range=(0.8, 1.6),
#         pad_val=114.0),
#     dict(
#         type='PhotoMetricDistortion',
#         brightness_delta=32,
#         contrast_range=(0.5, 1.5),
#         saturation_range=(0.5, 1.5),
#         hue_delta=18),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Resize', keep_ratio=True),
#     dict(type='Pad', pad_to_square=True, pad_val=114.0),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),

    # dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
>>>>>>> d1f949db84ce24686ad482744a0489c3878c7d53
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
<<<<<<< HEAD
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
=======
    # dict(
    #     type='MixUp',
    #     img_scale=img_scale,
    #     ratio_range=(0.8, 1.6),
    #     pad_val=114.0),
>>>>>>> d1f949db84ce24686ad482744a0489c3878c7d53
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
<<<<<<< HEAD
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=114.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
=======
    dict(type='Resize', img_scale=(320, 320), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
>>>>>>> d1f949db84ce24686ad482744a0489c3878c7d53
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
<<<<<<< HEAD
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=(300, 300), pad_val=114.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
=======
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
>>>>>>> d1f949db84ce24686ad482744a0489c3878c7d53
        ])
]

# train_dataset = dict(pipeline=train_pipeline)
# test_dataset = dict(pipeline=test_pipeline)
data_root = '/home/USR/active_learning_od/nutshell_mnist/'
# train_dataset = dict(
#     type='MultiImageMixDataset',
#     dataset=dict(
#         type='ALDetection',
#         # classes=[str(x) for x in range(10)],
#         ann_file=data_root + 'train/annotations/mnistdetection_train_instances.json',
#         img_prefix=data_root + 'train/img/',
#         pipeline=[
#             dict(type='LoadImageFromFile', to_float32=True),
#             dict(type='LoadAnnotations', with_bbox=True)
#         ],
#         filter_empty_gt=False,
#     ),
#     pipeline=train_pipeline,
#     dynamic_scale=img_scale
# )

classes = [str(x) for x in range(10)]
dataset_type = 'ALDetection'
data = dict(
    samples_per_gpu=4,
    # samples_per_gpu=32,
    workers_per_gpu=5,
    train=dict(
<<<<<<< HEAD
    type='MultiImageMixDataset',
    dataset=dict(
        type='ALDetection',
        classes=[str(x) for x in range(10)],
        ann_file=data_root + 'train/annotations/mnistdetection_train_instances.json',
        img_prefix=data_root + 'train/img/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline,
    dynamic_scale=img_scale
),
=======
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train/annotations/mnistdetection20k_train_instances.json',
        img_prefix=data_root + 'train/img/',
        pipeline=train_pipeline),
>>>>>>> d1f949db84ce24686ad482744a0489c3878c7d53
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
<<<<<<< HEAD
evaluation = dict(interval=1, metric='bbox')
=======

evaluation = dict(interval=500, metric='bbox')
>>>>>>> d1f949db84ce24686ad482744a0489c3878c7d53

# data = dict(
#     train=train_dataset,
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))

resume_from = None
interval = 10

# Execute in the order of insertion when the priority is the same.
# The smaller the value, the higher the priority
<<<<<<< HEAD
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
=======
# custom_hooks = [
#     dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
#     dict(
#         type='SyncRandomSizeHook',
#         ratio_range=(10, 20),
#         img_scale=img_scale,
#         priority=48),
#     dict(
#         type='SyncNormHook',
#         num_last_epochs=15,
#         interval=interval,
#         priority=48),
#     dict(type='ExpMomentumEMAHook', resume_from=resume_from, priority=49)
# ]
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]
checkpoint_config = dict(interval=interval)


# lr_config = dict(
#     _delete_=True,
#     policy='YOLOX',
#     warmup='exp',
#     by_epoch=False,
#     warmup_by_epoch=True,
#     warmup_ratio=1,
#     warmup_iters=5,  # 5 epoch
#     num_last_epochs=15,
#     min_lr_ratio=0.05)
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=7500)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.001,
    step=[5000, 6250])

checkpoint_config = dict(interval=runner['max_iters'])
log_config = dict(interval=50)

workflow = [('train', 1), ('val', 1)]

# optimizer = dict(
#     type='SGD',
#     lr=0.01,
#     momentum=0.9,
#     weight_decay=5e-4,
#     nesterov=True,
#     paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
>>>>>>> d1f949db84ce24686ad482744a0489c3878c7d53
