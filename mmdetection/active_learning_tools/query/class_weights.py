import os.path as osp
import json
import torch
import pandas as pd

weightings = [None, "class_wise"]


def compute_class_weights(config, metrics):
    if config.splits.query.weights in weightings:
        json_path = osp.join(config.data.train["ann_file"])
        json_dict = json.load(open(json_path, "r"))
        cats = [d["id"] for d in json_dict["categories"]]
        annotations = json_dict["annotations"]

        df = pd.DataFrame(columns=list(
            annotations[0].keys()), data=annotations)
        df["category_id"] = df["category_id"] - 1
        class_counts = torch.zeros(len(cats))
        count_series = df["category_id"].value_counts(sort=False)
        for i in range(len(cats)):
            if i in count_series.index:
                class_counts[i] = count_series[i]
        class_counts = class_counts.float()

        num_instances = len(annotations)
        num_classes = len(cats)

        return (num_instances + num_classes) / (class_counts + 1.)

    else:
        return torch.ones_like(metrics)
