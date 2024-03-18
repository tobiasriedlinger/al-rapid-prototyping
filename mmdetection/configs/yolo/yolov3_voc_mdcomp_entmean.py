checkpoint_config = dict(interval=18750)
log_config = dict(interval=500, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
dataset_id = 'voc'
dataset_type = 'ALDetection'
data_root = '/home/USR/mm-detection-active-learning-for-object-detection/checkpoints/voc_dataset_gt/'
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
classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type='ALDetection',
        classes=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                 'train', 'tvmonitor'),
        ann_file=
        '/home/USR/mm-detection-active-learning-for-object-detection/checkpoints/voc_dataset_gt/voc2007_train+voc2012_trainval_coco_fmt.json',
        img_prefix=
        '/home/USR/dataset_ground_truth/VOC_train',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Expand', mean=[0, 0, 0], to_rgb=True,
                ratio_range=(1, 2)),
            dict(
                type='MinIoURandomCrop',
                min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                min_crop_size=0.3),
            dict(
                type='Resize',
                img_scale=[(320, 320), (608, 608)],
                keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[0, 0, 0],
                std=[255.0, 255.0, 255.0],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='ALDetection',
        classes=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                 'train', 'tvmonitor'),
        ann_file=
        '/home/USR/mm-detection-active-learning-for-object-detection/checkpoints/voc_dataset_gt/voc2007_val_coco_fmt.json',
        img_prefix=
        '/home/USR/dataset_ground_truth/VOC_train',
        pipeline=[
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
        ]),
    test=dict(
        type='ALDetection',
        classes=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                 'train', 'tvmonitor'),
        ann_file=
        '/home/USR/mm-detection-active-learning-for-object-detection/checkpoints/voc_dataset_gt/voc2007_test_coco_fmt.json',
        img_prefix=
        '/home/dataset_sync/datasets/PASCAL_VOC/test/VOCdevkit/VOC2007/JPEGImages',
        pipeline=[
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
        ]))
evaluation = dict(interval=3000, metric=['bbox'])
model = dict(
    type='YOLOV3',
    backbone=dict(
        type='Darknet',
        depth=53,
        out_indices=(3, 4, 5),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/home/USR/mm-detection-active-learning-for-object-detection/checkpoints/yolov3/darknet_coco.pth'
        )),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=20,
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
        dropout=0.0),
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
    step=[11250, 15000])
runner = dict(type='IterBasedRunner', max_iters=18750)
splits = dict(
    initial=dict(size=200, ratio=None),
    validation=dict(size=200),
    train_track=dict(size=100),
    query=dict(
        pool_size=2000,
        method='entropy',
        distance='l2',
        aggregation='average',
        weights='class_wise',
        size=150,
        ratio=None,
        score_thr=0.5))
al_run = dict(run_count=0, initial_step=0, total_num_counts=15)
metrics = False
output_dir = '/home/USR/mm-detection-active-learning-for-object-detection/checkpoints/yolov3'
work_dir = './work_dirs/yolov3_voc_active_entropy'
