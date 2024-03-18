_base_ = [
    '../_base_/models/retinanet_r50_fpn.py', '../common/mstrain_3x_coco.py'
]
# optimizer
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

