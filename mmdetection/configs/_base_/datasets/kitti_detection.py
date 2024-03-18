# dataset settings
dataset_id = "kitti"
dataset_type = 'ALDetection'
data_root = '/home/USR/mm-detection-active-learning-for-object-detection/checkpoints/kitti_dataset_gt/'
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(type='Expand', mean=[0, 0, 0], to_rgb=True, ratio_range=(1, 2)),
    dict(type='MinIoURandomCrop',
         min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
         min_crop_size=0.3),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
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
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
classes = ["car", "van", "truck", "pedestrian",
           "person", "cyclist", "tram", "misc"]
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=10,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'kitti_train.json',
        img_prefix='/home/datasets/KITTI_tracking/training/image_02/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'kitti_val.json',
        img_prefix='/home/datasets/KITTI_tracking/training/image_02/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'kitti_val.json',
        img_prefix='/home/datasets/KITTI_tracking/training/image_02/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
