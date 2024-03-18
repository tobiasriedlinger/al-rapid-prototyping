_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/emnist_detection_active_mdcomp.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),

    ])
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=5, norm_type=2))

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
        num_classes=26,
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

# optimizer
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]

runner = dict(_delete_=True, type='IterBasedRunner', max_iters=25000)

evaluation = dict(interval=1000, metric='bbox')
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[20000, 22500])

checkpoint_config = dict(interval=runner['max_iters'])
workflow = [('train', 1), ('val', 1)]

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
        method="random",
        # insert aggregation method
        size=50,
        ratio=None,
        score_thr=0.1
    )
)

al_run = dict(
    run_count=0,
    initial_step=0,
    total_num_counts=42
)

metrics = False

work_dir = "/home/USR/mm-detection-active-learning-for-object-detection/checkpoints/retinanet/emnist_det_active/maxmap/0"
