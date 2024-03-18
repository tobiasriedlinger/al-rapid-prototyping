import os.path as osp

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

@DATASETS.register_module()
class KittiDataset(CustomDataset):
    """
    We train on all 8 classes that the KITTI data set offers.
    This tuple is used to convert the .txt-files to a COCO-style dictionary.
    """
    CLASSES = ("Car", "Van", "Truck", "Pedestrian", "Person", "Cyclist", "Tram", "Misc")

    def load_annotations(self, ann_file):
        """
        This method is used to generate a dictionary containing all ground truth information for the KITTI data set from the .txt-label files.
        :param ann_file: Text file containing image paths to all dataset images.
        :return data_infos: Dictionary structure containing all annotation information and image information needed for training.
        """
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}

        image_list = mmcv.list_from_file(self.ann_file)

        data_infos = []
        """
        Convert annotations to middle format, i.e. make it return a COCO-style dictionary
        """
        for image_id in image_list:
            filename = f"{self.img_prefix}/{image_id}.png"
            image = mmcv.imread(filename)
            height, width = image.shape[:2]

            data_info = dict(filename=f"{image_id}.png", width=width, height=height)

            # load annotations
            # if "image_2" in self.img_prefix:
            label_prefix = self.img_prefix.replace("image_2", "label_2")
            # else:
                # label_prefix = self.img_prefix.replace("images", "labels")
            lines = mmcv.list_from_file(osp.join(label_prefix, f"{image_id}.txt"))

            content = [line.strip().split(" ") for line in lines]
            bbox_names = [x[0] for x in content]
            bboxes = [[float(info) for info in x[4:8]] for x in content]

            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []

            for bbox_name, bbox in zip(bbox_names, bboxes):
                if bbox_name in cat2label:
                    gt_labels.append(cat2label[bbox_name])
                    gt_bboxes.append(bbox)
                else:
                    gt_labels_ignore.append(-1)
                    gt_bboxes_ignore.append(bbox)

            data_anno = dict(
                bboxes = np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels = np.array(gt_labels, dtype=np.long),
                bboxes_ignore=np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1, 4),
                labels_ignore=np.array(gt_labels_ignore, dtype=np.long)
            )

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos