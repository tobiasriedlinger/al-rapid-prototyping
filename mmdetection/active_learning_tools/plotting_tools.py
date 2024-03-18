import os
import os.path as osp
from tqdm import tqdm
import glob
import json
import torch
import numpy as np
import string

import mmcv

from mmdet.core.visualization.image import imshow_det_bboxes
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

class_names = {"kitti": ["car", "van", "truck", "pedestrian",
                         "person", "cyclist", "tram", "misc"],
               "mnist_det_active": [str(x) for x in range(10)],
               "emnist_det_active": list(string.ascii_uppercase),
               "voc": ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                       'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                       'tvmonitor'),
               "bdd": ("pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "trafficlight", "trafficsign")}

class_exceptions = {"kitti": ["person"],
                    "mnist_det_active": [],
                    "emnist_det_active": [],
                    "voc": [],
                    }
max_performance = {
    "retinanet":
        {
            "mnist_det_active": 0.9075,
            "emnist_det_active": 0.85575,
            "voc": 0.746
        },
    "retinanet_bs=4":
        {
            "mnist_det_active": 0.8670,
            "emnist_det_active": 0.8330
        },
    "yolov3":
        {
            "mnist_det_active": 0.965,
            "voc": 0.79425,
            "bdd": 0.419,
        },
    "faster_rcnn":
        {
            "voc": 0.79725
        }
}

mpl_colors = list(mcolors.TABLEAU_COLORS.values())
mpl_linestyles = ["solid", "dotted", "dashed", "dashdot"]

# def has_val_and_train_map(hist):
#     epochs = list(set([l["epoch"] for l in hist]))
#     restr_hist = [l for l in hist if (l["epoch"] == epochs[0])]
#     if len
#     return (len(restr_hist) == 4)


def plot_class_map_steps(path, dataset="mnist", base_metric="bbox_AP_50"):
    folder_list = glob.glob(osp.join(path, "step_*"))
    # print(folder_list)
    json_list = []
    for f in folder_list:
        json_path = osp.join(f, "test_map.json")
        if osp.exists(json_path):
            json_list.append(json.load(open(json_path, "r")))

    plt.clf()
    for i, c in enumerate(class_names[dataset]):
        if c in class_exceptions[dataset]:
            continue
        color = mpl_colors[i % len(mpl_colors)]
        ls = mpl_linestyles[i//len(mpl_colors)]
        # print([h[f"{base_metric}_{c}"] for h in json_list])
        plt.plot([h[f"{base_metric}_{c}"] for h in json_list],
                 c=color,
                 ls=ls,
                 label=f"{base_metric}_{c}")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.title("Test mAP class-wise")
    print(path+f"/class_wise_steps_{base_metric}_test.png")
    plt.savefig(path+f"/class_wise_steps_{base_metric}_test.png", dpi=600)


def plot_test_map(path, metric_id="bbox_mAP_50", store=False):
    folder_list = sorted(glob.glob(osp.join(path, "step_*")))
    json_list = []
    img_num_list = []
    anzahl = 0
    for folder_list2 in folder_list:
        for f in folder_list:
            if int(f.split('_')[-1]) == anzahl:
                json_path = osp.join(f, "test_map.json")
                data_path = osp.join(f, "labeled_data.json")

                if osp.exists(json_path):
                    json_list.append(json.load(open(json_path, "r")))
                    d = json.load(open(data_path, "r"))
                    img_num_list.append(len(d['ds_indices']))
                anzahl += 1
    # json.dump(json_list, osp.join(path, "test_map_over_steps.json"))
    map_list = [l[metric_id] for l in json_list]
    # if not store:
    #     plt.clf()
    plt.plot(img_num_list, map_list, label=path.split("/")[-2])
    path_slices = path.split("/")
    dataset = path_slices[-3]
    arch = path_slices[-4]
    if arch in max_performance and dataset in max_performance[arch] and not store:
        max_perf = max_performance[arch][dataset]
        plt.plot([0, img_num_list[-1]],
                 [max_perf, max_perf], color="gray", linewidth=2)
        plt.text(img_num_list[-1]+20, max_perf -
                 0.009, "$100 \\%$", color="gray")
        plt.plot([0, img_num_list[-1]],
                 [0.85*max_perf, 0.85*max_perf], color="green", linewidth=1, linestyle="--")
        plt.text(img_num_list[-1]+20, 0.85*max_perf -
                 0.009, "$85 \\%$", color="green")
        plt.text(img_num_list[-1]+20, 0.90*max_perf -
                 0.009, "$90 \\%$", color="orange")
        plt.text(img_num_list[-1]+20, 0.95*max_perf -
                 0.009, "$95 \\%$", color="red")
        plt.plot([0, img_num_list[-1]],
                 [0.9*max_perf, 0.9*max_perf], color="orange", linewidth=1, linestyle="--")
        plt.plot([0, img_num_list[-1]],
                 [0.95*max_perf, 0.95*max_perf], color="red", linewidth=1, linestyle="--")
    plt.xlabel("Image count")
    plt.grid()
    plt.ylabel(f"Test [{metric_id}]")
    if not store:
        plt.title(f"{arch} / {dataset} / {path_slices[-2]}")
        plt.legend()
        plt.ylim(min(map_list)-0.05, max_perf+0.05)
        plt.savefig(osp.join(path, f"test_{metric_id}.png"))
        print("Saved plot at ", osp.join(path, f"test_{metric_id}.png"))
        plt.clf()


def plot_class_map(path, dataset="mnist", base_metric="bbox_AP_50", smoothing=7):
    json_list = glob.glob(osp.join(path, "*.log.json"))

    hist = []
    # print(json_list)
    for line in open(json_list[-1], "r"):
        hist.append(json.loads(line))
    train_epochs = [h["epoch"]
                    for h in hist if "mode" in h.keys() and h["mode"] == "train"]
    max_epoch = train_epochs[-1]
    eval_interval = int(hist[0]["config"].split(
        "\nevaluation = dict(interval=")[-1].split(", metric")[0])
    x_plot = [x for x in range(eval_interval, max_epoch, eval_interval)]
    if max_epoch not in x_plot:
        x_plot.append(max_epoch)

    plt.clf()
    for i, c in enumerate(class_names[dataset]):
        if c in class_exceptions[dataset]:
            continue

        color = mpl_colors[
            i % len(mpl_colors)]
        ls = mpl_linestyles[i//len(mpl_colors)]
        count = 0
        train_map = []
        val_map = []
        class_metric = f"{base_metric}_{c}"
        for l in hist[1:]:
            if l["mode"] != "train":
                if class_metric in list(l.keys()):
                    if count % 2 == 0:
                        val_map.append(l[class_metric])
                    else:
                        train_map.append(l[class_metric])
                    count += 1
        # plt.plot([5*x for x in range(60)], val_losses, label="Val Loss")
        plt.plot(#x_plot,
                 val_map, alpha=0.2, c=color)
        
        val_map += [val_map[-1] for _ in range(smoothing)]
        mean_val_map = [np.mean(val_map[i:i+smoothing])
                        for i in range(len(val_map)-smoothing)]
        
        plt.plot(x_plot, mean_val_map,
                 label=f"{class_metric}", c=color, ls=ls)
        # plt.plot(train_map, label=f"Train Loss [{loss_id}]")
    plt.legend(ncol=2)
    # plt.yscale("log")
    plt.grid(True)
    plt.title("Validation Performance")
    # plt.legend()
    plt.savefig(path+f"class_wise_{base_metric}.png", dpi=600)


def plot_loss_curve(path, loss_id="loss", maxmap=False):
    # print(path)
    json_list = glob.glob(osp.join(path, "*.log.json"))

    hist = []
    print(json_list)
    for line in open(json_list[-1], "r"):
        hist.append(json.loads(line))
        # print(json.loads(line))
    # print(hist)
    # train_epochs = [h["epoch"]
    #                 for h in hist if "mode" in h.keys() and h["mode"] == "train"]
    train_epochs = [h["iter"]
                    for h in hist if "mode" in h.keys() and h["mode"] == "train"]
    max_epoch = train_epochs[-1]
    # max_iter_list = [h for h in hist if h["epoch"] == max_epoch]
    # max_iter = max_iter_list[-1]["iter"]
    train_losses = []
    val_losses = []
    train_map = []
    val_map = []
    max_iter = 0
    count = 0
    eval_interval = int(hist[0]["config"].split(
        "\nevaluation = dict(interval=")[-1].split(", metric")[0])
    steps = [int(x) for x in hist[0]["config"].split("lr_config")[1].split("step=[")[1].split("])")[0].split(", ")]
    for l in hist[1:]:
        # print(l)
        if l["mode"] == "train":
            train_losses.append(l[loss_id])
            # max_iter = max(int(max_iter), int(l["iter"]/10))
            max_iter = max(int(max_iter), int(l["iter"]))
        else:
            if "bbox_mAP_50" in list(l.keys()):
                if count % 2 == 1 and not maxmap:
                    train_map.append(l["bbox_mAP_50"])
                else:
                    val_map.append(l["bbox_mAP_50"])
                count += 1
            elif loss_id in list(l.keys()):
                val_losses.append(l[loss_id])
                

    # name_list = ["Train Loss", "Val Loss", "Train mAP"]
    # id_list = ["train_loss", "val_loss", "train_map"]
    # fig_path = "/home/riedlinger/active_learning_od/mmdetection/checkpoints/yolov3/mnist_det/random/run_0/step_0/"
    plt.clf()
    # plt.plot([5*x for x in range(60)], val_losses, label="Val Loss")
    plot_x = [max_iter*x for x in range(max_epoch)][:len(val_losses)]
    if val_losses:
        plt.plot(plot_x, val_losses,
                 label=f"Val Loss [{loss_id}]")
    plt.plot(train_losses, label=f"Train Loss [{loss_id}]")
    for x in steps:
        plt.axvline(x=x, color="k")
    plt.legend()
    plt.yscale("log")
    plt.grid(True)
    plt.savefig(path+f"{loss_id}.png")
    plt.clf()
    x_plot = [x for x in range(eval_interval, max_epoch, eval_interval)]
    if max_epoch not in x_plot:
        x_plot.append(max_epoch)
    if val_map:
        # print(max_epoch, val_map)
        # print([x for x in range(eval_interval, max_epoch, eval_interval)])
        if len(x_plot[:len(val_map)]) != len(val_map):
            eval_interval = 250
            x_plot = [x for x in range(eval_interval, max_epoch+1, eval_interval)]
        plt.plot(x_plot[:len(val_map)],
                 val_map, label="Val mAP")
    plt.plot(x_plot[:len(train_map)],
             train_map, label="Train mAP")
    plt.grid(True)
    for x in steps:
        plt.axvline(x=x, color="k")
    plt.legend()
    plt.savefig(path+"map.png")
    plt.clf()


def plot_query(test_path, dataset, score_id="weighted_entropies"):
    target_path = osp.join(test_path, f"query_imgs_{score_id}")
    os.makedirs(target_path, exist_ok=True)

    unlabeled_d = json.load(
        open(osp.join(test_path, "unlabeled_data.json"), "r"))
    query_output_d = json.load(
        open(osp.join(test_path, "query_output.json"), "r"))
    # print(query_output_d)

    inds = query_output_d['queried_ids']
    for i, id in enumerate(inds[:20]):
        img_id = query_output_d['images'].index(id)
        # print(img_id, id)
        for d in unlabeled_d['images']:
            if d['id'] == id:
                file_name = d['file_name']
                break
        path = osp.join(query_output_d['img_prefix'], file_name)
        img = mmcv.imread(path)
        img = mmcv.image.imrescale(img, (1000, 600))
        # img = mmcv.image.imresize(img, (1000, 600))
        # img = mmcv.image.imresize(img, (300, 300))

        bboxes = np.array(query_output_d['detections'][str(img_id)])
        bboxes[:, 4] = np.array(query_output_d[score_id][str(img_id)])
        labels = np.array(query_output_d['labels'][str(img_id)])
        query_score = query_output_d['query_scores'][i]

        img = imshow_det_bboxes(img,
                                bboxes,
                                labels,
                                class_names=class_names[dataset],
                                out_file=osp.join(
                                    target_path, "{}_{:.3f}_".format(i, query_score) + file_name),
                                thickness=1,
                                show=False
                                )
        
def plot_ground_truth(label_path, img_path, num_images, dataset, target_path="/home/riedlinger"):
    labels = json.load(open(label_path, "r"))
    
    for d in labels["images"][:num_images]:
        idx = d["id"]
        file_path = osp.join(img_path, d["file_name"])
        
        img = mmcv.imread(file_path)
        anns = [x for x in labels["annotations"] if x["image_id"] == idx]
        print(idx, file_path, anns)
        mmcv.mkdir_or_exist(target_path)
        bboxes = np.array([x["bbox"] for x in anns])
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        img = imshow_det_bboxes(img,
                                bboxes,
                                np.array([x["category_id"]-1 for x in anns]),
                                class_names=class_names[dataset],
                                out_file=osp.join(target_path, d["file_name"]),
                                thickness=1
                                )


if __name__ == "__main__":
    # dataset = "kitti"
    dataset = "voc"
    target_path = "/home/riedlinger/al_nutshell/mm-detection-active-learning-for-object-detection/results/al_runs/frcnn/voc/entropy/run_0/step_8"
    # target_path = "/home/riedlinger/al_nutshell/mm-detection-active-learning-for-object-detection/checkpoints/kitti_dataset_gt/plots"
    # plot_ground_truth("/home/riedlinger/al_nutshell/mm-detection-active-learning-for-object-detection/checkpoints/kitti_dataset_gt/kitti_train.json", '/home/datasets/KITTI_tracking/training/image_02/', 20, dataset, target_path)
    # plot_ground_truth("/home/riedlinger/al_nutshell/mm-detection-active-learning-for-object-detection/checkpoints/voc_dataset_gt/voc2007_test_coco_fmt.json", '/home/datasets/PASCAL_VOC/test/VOCdevkit/VOC2007/JPEGImages', 20, dataset, target_path)
    # 20211119_113323.log.json"
    # dataset = "emnist_det_active"
    # dataset = "emnist_det_corrupt_10"
    # dataset = "mnist_det_active"
    # dataset = "mnist"
    # dataset = "emnist_det"
    # dataset = "voc"
    # dataset = "bdd"
    # dataset = "bisec/bdd"
    # method = "entropy"
    method="random"
    # method="meta_detect"
    # method = "prob_margin"
    # method = "dropout"
    # method = "entropy"
    # model = "retinanet_bs=4"
    # model = "retinanet"
    # model = "yolov3"
    # model = "frcnn18"
    model = "yolov3_dn20"
    maxmap = False

    run = 900
    # methods = ["entropy"]
    # for m in methods:
    # test_path = f"/home/riedlinger/al_nutshell/mm-detection-active-learning-for-object-detection/checkpoints/{model}/{dataset}/{method}/run_{run}"
    test_path = f"/home/riedlinger/al_nutshell/results_aggr/al_runs/{model}/{dataset}/{method}/run_{run}"
    # test_path = f"/home/riedlinger/al_nutshell/results/al_runs/{model}/{dataset}/{method}/run_{run}"
    # test_path = f"/home/schubert/MMDetection-AL/mm-detection-active-learning-for-object-detection/checkpoints/{model}/{dataset}/{method}/run_{run}"
    # test_path = "/home/riedlinger/al_nutshell/results/al_runs/retinanet_bs=4/emnist_det_active/random/run_0"
    # test_path = "/home/schubert/MMDetection-AL/mm-detection-active-learning-for-object-detection/checkpoints/retinanet/mnist_det_active/random/run_0/"
    # for i in tqdm(range(1, 9)):
    #     p = f"{test_path}/step_{i}/"
    # p = f"{test_path}/step_0/"
    # for i in range(1):
    #     p = f"{test_path}/step_{i}/"
    #     # plot_query(p, dataset, score_id="weighted_meta_detect")
    #     plot_loss_curve(p, maxmap=False)
    # plot_test_map(test_path)
        # plot_loss_curve(p, "loss_cls")
    p = "/home/riedlinger/al_nutshell/results_aggr/maxmap/yolov3/voc/bb_ckpt"
    p = "/home/riedlinger/al_nutshell/results_aggr/al_runs/retinanet/voc/entropy/run_0/step_3"
    # p = "/home/riedlinger/al_nutshell/results_aggr/al_runs/frcnn/voc/entropy/run_3/step_1"
    # p = "/home/riedlinger/al_nutshell/results_aggr/al_runs/retinanet/bdd/entropy/run_3/step_1"
    plot_query(p, "voc", score_id="top_scores_post_nms")
    # plot_query(p, "voc", score_id="weighted_entropies")
    # p = "/home/riedlinger/al_nutshell/results_aggr/maxmap/yolov3/mnist_det_active/0"
    # plot_loss_curve(p, maxmap=False)
    # p = "/home/riedlinger/al_nutshell/results_aggr/al_runs/retinanet/mnist_det_active/random/run_500/step_0"
    # plot_loss_curve(p, maxmap=False)
    p = "/home/riedlinger/al_nutshell/results_aggr/maxmap/yolov3/voc/bb_ckpt_300k"
    plot_loss_curve(p, maxmap=True)
    p = "/home/riedlinger/al_nutshell/results_aggr/al_runs/retinanet/voc/random/run_0/step_0"
    plot_loss_curve(p, maxmap=False)
        # plot_loss_curve(p, "loss_rpn_cls")
        # plot_loss_curve(p, "loss_rpn_bbox")
        # plot_loss_curve(p, "loss_conf")
        # plot_loss_curve(p, "loss_xy")
        # plot_loss_curve(p, "loss_wh")
        # plot_loss_curve(p, "loss_bbox")
        # plot_class_map(p, dataset=dataset)
        # plot_test_map(test_path, store=True)
    # plot_class_map_steps(test_path, dataset=dataset)
    # method = "random"
    # test_path = f"/home/riedlinger/al_nutshell/mm-detection-active-learning-for-object-detection/checkpoints/{model}/{dataset}/{method}/run_{run}"
    # test_path = f"/home/schubert/MMDetection-AL/mm-detection-active-learning-for-object-detection/checkpoints/{model}/{dataset}/{method}/run_{run}"
    # plot_test_map(test_path, store=False)
    # for i in tqdm(range(19)):
    #         p = f"{test_path}/step_{i}/"
    #         plot_loss_curve(p)
    # plot_test_map(test_path)
