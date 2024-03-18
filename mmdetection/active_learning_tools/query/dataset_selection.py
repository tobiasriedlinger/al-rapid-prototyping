import json
import numpy as np
import random


def split_random_dataset(annotation_dict, train_count=200, seed=0, labeled_ids=[]):
    image_list = annotation_dict["images"]
    index_range = list([ann["id"] for ann in annotation_dict["images"]])

    if not labeled_ids:
        random.seed(seed)
        random.shuffle(index_range)
        labeled_indices = index_range[:train_count]
        unlabeled_indices = index_range[train_count:]
    else:
        labeled_indices = labeled_ids
        random.seed(seed)
        random.shuffle(index_range)
        unlabeled_indices = [x for x in index_range if x not in labeled_ids]
    labeled_flags = [int(i in labeled_indices) for i in index_range]
    unlabeled_flags = [int(i in unlabeled_indices)
                       for i in index_range]
    labeled_images = [
        img for img in image_list if img["id"] in labeled_indices]
    unlabeled_images = [
        img for img in image_list if img["id"] in unlabeled_indices]

    labeled = {"ds_indices": labeled_indices,
               "ds_flags": labeled_flags, "images": labeled_images}
    unlabeled = {"ds_indices": unlabeled_indices,
                 "ds_flags": unlabeled_flags, "images": unlabeled_images}

    return labeled, unlabeled


def build_annotations_from_ids(cfg, image_ids=None, full_annotations=None):
    if full_annotations is None:
        full_annotations = json.load(open(cfg["ann_file"], "r"))
    if image_ids is None:
        return full_annotations
    else:
        image_list = [ann for ann in full_annotations["images"]
                      if ann["id"] in image_ids]
        annotation_list = [ann for ann in full_annotations["annotations"]
                           if ann["image_id"] in [i["id"] for i in image_list]]
        return {
            "info": full_annotations["info"],
            "licenses": full_annotations["licenses"],
            "images": image_list,
            "annotations": annotation_list,
            "categories": full_annotations["categories"]
        }
