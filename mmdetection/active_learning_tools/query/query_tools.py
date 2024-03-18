import os.path as osp
import copy
import time
import torch
import mmcv
import numpy as np

from mmdet.datasets import build_dataset, build_dataloader
from active_learning_tools.query.dataset_selection import build_annotations_from_ids, split_random_dataset

from active_learning_tools.query.entropy import entropy_query
from active_learning_tools.query.probability_margin import prob_margin_query
from active_learning_tools.query.mc_dropout import dropout_query
from active_learning_tools.query.mutual_information import mutual_information_query
from active_learning_tools.query.meta_detect import meta_detect_query
from active_learning_tools.query.core_set import core_set_query
from active_learning_tools.query.log_triggers import log_triggers_query

uncertainty_queries = {"entropy": 0,
                       "prob_margin": 0,
                       "dropout": 0,
                       "mutual_information": 0,
                       "meta_detect": 0,
                       "log_triggers": 0,
                       }

density_queries = {"core": 0,
                   "kmeans": 0}

image_aggregations = {"sum": torch.sum,
                      "average": torch.mean,
                      "maximum": torch.max}


def query_data(model,
               labeled,
               unlabeled,
               full_annotations,
               cfg,
               base_target_path="./",
               step=0,
               val_ann_file_orig=None,
               timing_dict={}
               ):
    print("Starting query...")
    tic = time.time()
    if cfg.splits.query.size:
        query_size = cfg.splits.query.size
    else:
        query_size = int(cfg.splits.query.ratio *
                         len(full_annotations["images"]))

    if cfg.splits.query.method == "random":
        unlabeled_anns = build_annotations_from_ids(cfg.data.train,
                                                    image_ids=unlabeled["ds_indices"][:cfg.splits.query.pool_size],
                                                    full_annotations=full_annotations)
        mmcv.dump(unlabeled_anns, osp.join(
            base_target_path, "unlabeled_anns.json"))
        timing_dict.update({"query_inference": 0.0})
        tic = time.time()
        labeled_add, unlabeled = split_random_dataset(unlabeled_anns,
                                                      train_count=query_size,
                                                      seed=cfg.al_run.run_count)
        timing_dict.update({"query_construction": float(time.time() - tic)})
        mmcv.dump({"queried_ids": labeled_add["ds_indices"]},
                      osp.join(base_target_path,
                               f"step_{step}/query_output.json"))
        labeled, unlabeled = split_random_dataset(annotation_dict=full_annotations,
                                                      labeled_ids=labeled["ds_indices"]+labeled_add["ds_indices"])
    else:
        unlabeled_anns = build_annotations_from_ids(cfg.data.train,
                                                    image_ids=unlabeled["ds_indices"][:cfg.splits.query.pool_size],
                                                    full_annotations=full_annotations)
        mmcv.dump(unlabeled_anns, osp.join(
            base_target_path, "unlabeled_anns.json"))

        output_metrics = []
        if cfg.splits.query.method in list(uncertainty_queries.keys()):
            query_dict = copy.deepcopy(cfg.data.test)
            query_dict["ann_file"] = osp.join(
                base_target_path, "unlabeled_anns.json")
            query_dict["img_prefix"] = cfg.data.train["img_prefix"]
            query_dataset = build_dataset(query_dict)
            query_dataloader = build_dataloader(query_dataset,
                                                samples_per_gpu=1,
                                                workers_per_gpu=1,
                                                dist=False,
                                                shuffle=False)
            
            if cfg.splits.query.method == "entropy":
                query_ids, output_metrics, timing_dict = entropy_query(model,
                                                          query_dataloader,
                                                          cfg.model,
                                                          post_nms_topk=20,
                                                          image_aggregation=image_aggregations[
                                                              cfg.splits.query.aggregation],
                                                          img_ids=query_dataset.img_ids,
                                                          query_score_thr=cfg.splits.query.score_thr,
                                                          query_size=cfg.splits.query.size,
                                                          config=cfg,
                                                          timing_dict=timing_dict)
            elif cfg.splits.query.method == "prob_margin":
                query_ids, output_metrics, timing_dict = prob_margin_query(model,
                                                              query_dataloader,
                                                              cfg.model,
                                                              post_nms_topk=20,
                                                              image_aggregation=image_aggregations[
                                                                  cfg.splits.query.aggregation],
                                                              img_ids=query_dataset.img_ids,
                                                              query_score_thr=cfg.splits.query.score_thr,
                                                              query_size=cfg.splits.query.size,
                                                              config=cfg,
                                                              timing_dict=timing_dict)
            elif cfg.splits.query.method == "dropout":
                query_ids, output_metrics, timing_dict = dropout_query(model,
                                                          query_dataloader,
                                                              cfg.model,
                                                              post_nms_topk=20,
                                                              image_aggregation=image_aggregations[
                                                                  cfg.splits.query.aggregation],
                                                              img_ids=query_dataset.img_ids,
                                                              query_score_thr=cfg.splits.query.score_thr,
                                                              query_size=cfg.splits.query.size,
                                                              config=cfg,
                                                              timing_dict=timing_dict)
            elif cfg.splits.query.method == "mutual_information":
                query_ids, output_metrics, timing_dict = mutual_information_query(model,
                                                              query_dataloader,
                                                              cfg.model,
                                                              post_nms_topk=20,
                                                              image_aggregation=image_aggregations[
                                                                  cfg.splits.query.aggregation],
                                                              img_ids=query_dataset.img_ids,
                                                              query_score_thr=cfg.splits.query.score_thr,
                                                              query_size=cfg.splits.query.size,
                                                              config=cfg,
                                                              timing_dict=timing_dict)
            elif cfg.splits.query.method == "meta_detect":
                val_ann_file_train = cfg.data.val["ann_file"]
                cfg.data.val["ann_file"] = val_ann_file_orig
                cfg.data.val['pipeline'] = cfg.test_pipeline
                val_dataset = build_dataset(cfg.data.val)
                val_dataloader = build_dataloader(val_dataset,
                                                samples_per_gpu=1,
                                                workers_per_gpu=1,
                                                dist=False,
                                                shuffle=False)
                query_ids, output_metrics = meta_detect_query(model,
                                                              query_dataloader,
                                                              val_dataloader,
                                                              cfg.model,
                                                              post_nms_topk=20,
                                                              image_aggregation=image_aggregations[
                                                                  cfg.splits.query.aggregation],
                                                              img_ids=query_dataset.img_ids,
                                                              query_score_thr=cfg.splits.query.score_thr,
                                                              query_size=cfg.splits.query.size,
                                                              config=cfg, 
                                                              base_target_path=base_target_path,
                                                              step=step,
                                                              val_dict=cfg.data.val['ann_file'],
                                                              query_dict=query_dict['ann_file'])
                cfg.data.val["ann_file"] = val_ann_file_train
            elif cfg.splits.query.method == "log_triggers":
                query_ids, output_metrics, timing_dict = log_triggers_query(model,
                                                          query_dataloader,
                                                          cfg.model,
                                                          post_nms_topk=20,
                                                          image_aggregation=image_aggregations[
                                                              cfg.splits.query.aggregation],
                                                          img_ids=query_dataset.img_ids,
                                                          query_score_thr=cfg.splits.query.score_thr,
                                                          query_size=cfg.splits.query.size,
                                                          config=cfg,
                                                          timing_dict=timing_dict)
            labeled, unlabeled = split_random_dataset(annotation_dict=full_annotations,
                                                      labeled_ids=labeled["ds_indices"]+query_ids)
            if 'metrics' in cfg.model.test_cfg:
                cfg.model.test_cfg["metrics"] = False
            else:
                cfg.model.test_cfg.rcnn["metrics"] = False
            output_metrics.update({"images": query_dataset.img_ids,
                                   "img_prefix": query_dataset.img_prefix})
            mmcv.dump(output_metrics,
                      osp.join(base_target_path,
                               f"step_{step}/query_output.json"))
        elif cfg.splits.query.method in list(density_queries.keys()):
            query_dict = copy.deepcopy(cfg.data.test)
            query_dict["ann_file"] = osp.join(
                base_target_path, "unlabeled_anns.json")
            query_dict["img_prefix"] = cfg.data.train["img_prefix"]
            query_dataset = build_dataset(query_dict)
            query_dataloader = build_dataloader(query_dataset,
                                                samples_per_gpu=1,
                                                workers_per_gpu=1,
                                                dist=False,
                                                shuffle=False)
            model.eval()

            annotation_path = osp.join(base_target_path, f"step_{step}")
            query_ids, output_metrics, timing_dict = core_set_query(model, 
                                                       query_dataloader, 
                                                       img_ids=query_dataset.img_ids,
                                                       query_size=query_size,
                                                       config=cfg,
                                                       annotation_path=annotation_path,
                                                       timing_dict=timing_dict)

            labeled, unlabeled = split_random_dataset(annotation_dict=full_annotations,
                                                      labeled_ids=labeled["ds_indices"]+query_ids)
            output_metrics.update({"images": query_dataset.img_ids,
                                   "img_prefix": query_dataset.img_prefix})
            mmcv.dump(output_metrics,
                      osp.join(base_target_path,
                               f"step_{step}/query_output.json"))

    print("Query finished after ", time.time() - tic, " seconds.")

    return labeled, unlabeled, timing_dict
