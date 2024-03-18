import os.path as osp
import string
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class_names = {"kitti": ["car", "van", "truck", "pedestrian",
                         "person", "cyclist", "tram", "misc"],
               "mnist_det": list(range(10)),
               "emnist_det": list(string.ascii_uppercase)}


def build_class_statistics(json_path, plot=False, dataset="mnist_det"):
    json_dict = json.load(open(json_path, "r"))
    cats = [d["id"] for d in json_dict["categories"]]
    images = [d["id"] for d in json_dict["images"]]
    annotations = json_dict["annotations"]

    df = pd.DataFrame(columns=list(annotations[0].keys()), data=annotations)
    df["category_id"] = df["category_id"] - 1

    path_parts = json_path.split("/")
    dataset = path_parts[-5]
    method = path_parts[-4]
    run = int(path_parts[-3].split("run_")[-1])
    step = int(path_parts[-2].split("step_")[-1])
    save_path = osp.join(*path_parts[:-1])
    df.to_csv(f"/{save_path}/labeled_boxes.csv")

    if plot:
        sns.histplot(data=df,
                     x="category_id",
                     bins=len(cats),
                     discrete=True,
                     alpha=0.2
                     )
        cat_labels = class_names[dataset]
        xtick_rotation = "vertical" if (
            max([len(str(c)) for c in cats]) > 3) else None
        plt.xticks(range(len(cats)), cat_labels, rotation=xtick_rotation)
        title_str = "{}, {}, Run {}, Step {}, #Images {}, #Boxes {}".format(dataset,
                                                                            method,
                                                                            run,
                                                                            step,
                                                                            len(images),
                                                                            len(annotations))
        plt.title(title_str, fontsize=12)
        plt.savefig(f"/{save_path}/labeled_boxes.png", dpi=600)

    return df

###
#   Implement plotting routine for query boxes.
###


def plot_query_boxes(json_path):
    query_output = json.load(open(json_path, "r"))
    pass


if __name__ == "__main__":
    dataset = "emnist_det"
    method = "random"
    run = 0

    for step in range(3):
        print("Step: ", step)
        p = f"/home/USR/active_learning_od/mmdetection/checkpoints/yolov3/{dataset}/{method}/run_{run}/step_{step}/labeled_anns.json"
        build_class_statistics(p, dataset=dataset, plot=True)
