_base_ = ['../_base_/default_runtime.py',
          '../_base_/datasets/emnist_detection_active_mdcomp.py']
# model settings
model = dict(
    type='YOLOV3',
    backbone=dict(
        type='Darknet',
        depth=20,
        out_indices=(3, 4, 5),
        # init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53')),
        init_cfg=dict(type='Pretrained', checkpoint='/home/USR/mm-detection-active-learning-for-object-detection/checkpoints/yolov3/darknet20_coco.pth')),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[512, 256, 128],
        out_channels=[256, 128, 64]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=26,
        in_channels=[256, 128, 64],
        out_channels=[512, 256, 128],
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
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.0,
        conf_thr=0.0,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100,
        metrics=False))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.1,
    step=[30000, 32500])
runner = dict(type='IterBasedRunner', max_iters=35000)
evaluation = dict(interval=500, metric=['bbox'])
checkpoint_config = dict(interval=runner["max_iters"])
workflow = [('train', 1) , ('val', 1)]

# Active Learning Settings
splits = dict(
    initial=dict(
        size=100,
        ratio=None
    ),
    validation=dict(
        size=250
    ),
    train_track=dict(
        size=250
    ),
    query=dict(
        pool_size=2000,
        method='entropy',
        n_samples=10,
        distance='l2',
        aggregation='maximum',
        weights='class_wise',
        size=100,
        ratio=None,
        score_thr=0.5)
)

al_run = dict(
    run_count=0,
    initial_step=0,
    total_num_counts=16
)

metrics = False

output_dir = "/home/USR/results_aggr/al_runs/yolov3_dn20"
