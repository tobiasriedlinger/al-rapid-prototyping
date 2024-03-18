_base_ = ['../_base_/datasets/bdd_jue.py',
          '../_base_/default_runtime.py']

checkpoint_config = dict(interval=55000)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
runner = dict(type='IterBasedRunner', max_iters=55000)
evaluation = dict(interval=2500, metric=['bbox'])

dataset_id = 'bdd'
dataset_type = 'ALDetection'
data_root = '/home/USR/dataset_ground_truth/bdd100k/'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(type='Expand', mean=[0, 0, 0], to_rgb=True, ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=[(320, 320), (608, 608)], keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[0, 0, 0],
        std=[255.0, 255.0, 255.0],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[0, 0, 0],
                std=[255.0, 255.0, 255.0],
                to_rgb=True),
            dict(type='Pad', pad_to_square=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

classes = ("pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "trafficlight", "trafficsign")
data = dict(
    
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'bdd100k_train_annotations_clear_daytime.json',
        img_prefix='/home/USR/.datasets/bdd100k/images/100k/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,

        ann_file=data_root + 'bdd100k_val_annotations_clear_daytime.json',
        img_prefix='/home/USR/.datasets/bdd100k/images/100k/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,

        ann_file=data_root + 'bdd100k_test_annotations_clear_daytime.json',
        img_prefix='/home/USR/.datasets/bdd100k/images/100k/val',
        pipeline=test_pipeline))

model = dict(
    type='YOLOV3',
    backbone=dict(
        type='Darknet',
        depth=53,
        out_indices=(3, 4, 5),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/home/USR/active_learning/mm-detection-active-learning-for-object-detection/checkpoints/yolov3/darknet_coco.pth'
        )),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=10,
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0),
        dropout=0.5),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.01,
        conf_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100,
        metrics=False))
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.1,
    step=[35000, 50000])
splits = dict(
    initial=dict(size=300, ratio=None),
    validation=dict(size=500),
    train_track=dict(size=500),
    query=dict(
        pool_size=2000,
        method='mutual_information',
        n_samples=10,
        distance='l2',
        aggregation='average',
        weights='class_wise',
        size=200,
        ratio=None,
        score_thr=0.5))
al_run = dict(run_count=0, initial_step=0, total_num_counts=3)
metrics = False
output_dir = '/home/USR/active_learning/results/al_runs/yolov3'
work_dir = '/home/USR/active_learning/results/al_runs/yolov3'
