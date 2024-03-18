# dataset settings
dataset_id = "bdd"
dataset_type = 'ALDetection'
data_root = '/home/USR/dataset_ground_truth/bdd100k/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
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
evaluation = dict(interval=1, metric='bbox')
