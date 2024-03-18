import torch
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
import xgboost as xgb
import json
import math

from mmcv.parallel import MMDataParallel

from mmdet.apis.test import single_gpu_tensor_outputs
from mmdet.models import TwoStageDetector

from mmdet.core.post_processing.bbox_nms import multiclass_nms
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from active_learning_tools.query.class_weights import compute_class_weights

META_DETECT_METRICS = ['Number of Candidate Boxes', 'x_min',
                  'x_max', 'x_mean', 'x_std', 'y_min', 'y_max', 'y_mean', 'y_std', 'w_min', 'w_max', 'w_mean', 'w_std',
                  'h_min', 'h_max', 'h_mean', 'h_std', 'size', 'size_min', 'size_max', 'size_mean', 'size_std',
                  'circum', 'circum_min', 'circum_max', 'circum_mean', 'circum_std', 'size/circum', 'size/circum_min',
                  'size/circum_max', 'size/circum_mean', 'size/circum_std', 'score_min', 'score_max', 'score_mean', 'score_std',
                  'IoU_pb_min', 'IoU_pb_max', 'IoU_pb_mean', 'IoU_pb_std']

def metric_calculation(dets, labels, top_boxes, top_probas, nms_cfg, score_factors=None, query_score_thr=0.0, gt_bboxes_img=[], gt_labels_img=[], cal_iou=False):
    if score_factors is not None:
        score_factors = np.asarray(score_factors.cpu())[0][:]
        top_probas = np.asarray(top_probas) * score_factors[:, np.newaxis]
    top_labels = np.argmax(top_probas[:,:-1], axis=1)
    ious = bbox_overlaps(torch.FloatTensor(dets[:, :-1]),
                         torch.FloatTensor(top_boxes), 
                         mode='iou')

    top_boxes = np.append(top_boxes, np.asmatrix(np.max(top_probas[:,:-1], axis=1)).T, axis=1)
            
    ious = np.asarray(ious)
    labels = np.asarray(labels)
    ious = [np.where(top_probas[:, labels[0][idx]] >= query_score_thr, i, 0) for idx, i in enumerate(ious)]
    metrics_single_image = np.zeros((len(ious), 40))
    for i in range(len(ious)):
        candidates = top_boxes[ious[i][:] >= nms_cfg.iou_threshold]
        candidates = np.concatenate((candidates, np.asmatrix(ious[i][ious[i][:] >= nms_cfg.iou_threshold]).T), axis=1)
        minimum = 0
        maximum = 0
        assignment_bboxes = np.zeros((candidates.shape[0], 9))
        for j in range(candidates.shape[0]):
            assignment_bboxes[j, 0] = 0.5 * (candidates[j, 0] + candidates[j, 2])
            assignment_bboxes[j, 1] = 0.5 * (candidates[j, 1] + candidates[j, 3])
            assignment_bboxes[j, 2] = abs(0.5 * (candidates[j, 0] - candidates[j, 2]))
            assignment_bboxes[j, 3] = abs(0.5 * (candidates[j, 1] - candidates[j, 3]))
            assignment_bboxes[j, 4] = candidates[j, 4]
            assignment_bboxes[j, 5] = candidates[j, 5]
            assignment_bboxes[j, 6] = (0.5 * ((0.5 * (candidates[j, 0] + candidates[j, 2])) - ( abs(0.5 * (candidates[j, 0] - candidates[j, 2]))))) * (0.5 * ((0.5 * (candidates[j, 1] + candidates[j, 3])) - (abs(0.5 * (candidates[j, 1] - candidates[j, 3])))))
            assignment_bboxes[j, 7] = (candidates[j, 2] - candidates[j, 0]) + (candidates[j, 3] - candidates[j, 1])
            assignment_bboxes[j, 8] = ((0.5 * (abs(candidates[j, 2] - candidates[j, 0]))) * (0.5 * (abs(candidates[j, 3] - candidates[j, 1])))) / ((abs(candidates[j, 2] - candidates[j, 0])) + (abs(candidates[j, 3] - candidates[j, 1])))

        iou = ((np.asarray(assignment_bboxes[:, 5])).flatten()).astype('float32')

        if len(iou) > 1:
            minimum = np.min(iou[iou > 0])
            maximum = np.max(iou[iou < 1])

        assignment_bboxes = np.asmatrix(assignment_bboxes.astype('float32'))
        
        metrics_single_image[i, :] = [assignment_bboxes.shape[0], float(np.min((assignment_bboxes[:, 0]), axis=0)),
                                    float(np.max((assignment_bboxes[:, 0]), axis=0)),
                                    float(np.mean((assignment_bboxes[:, 0]), axis=0)),
                                    float(np.std((assignment_bboxes[:, 0]), axis=0)),
                                    float(np.min((assignment_bboxes[:, 1]), axis=0)),
                                    float(np.max((assignment_bboxes[:, 1]), axis=0)),
                                    float(np.mean((assignment_bboxes[:, 1]), axis=0)),
                                    float(np.std((assignment_bboxes[:, 1]), axis=0)),
                                    float(np.min((assignment_bboxes[:, 2]), axis=0)),
                                    float(np.max((assignment_bboxes[:, 2]), axis=0)),
                                    float(np.mean((assignment_bboxes[:, 2]), axis=0)),
                                    float(np.std((assignment_bboxes[:, 2]), axis=0)),
                                    float(np.min((assignment_bboxes[:, 3]), axis=0)),
                                    float(np.max((assignment_bboxes[:, 3]), axis=0)),
                                    float(np.mean((assignment_bboxes[:, 3]), axis=0)),
                                    float(np.std((assignment_bboxes[:, 3]), axis=0)),
                                    float((0.5 * ((0.5 * (dets[i, 0] + dets[i, 2])) - (abs(0.5 * (dets[i, 0] - dets[i, 2]))))) * (0.5 * ((0.5 * (dets[i, 1] + dets[i, 3])) - (abs(0.5 * (dets[i, 1] - dets[i, 3])))))),
                                    float(np.min((assignment_bboxes[:, 6]), axis=0)),
                                    float(np.max((assignment_bboxes[:, 6]), axis=0)),
                                    float(np.mean((assignment_bboxes[:, 6]), axis=0)),
                                    float(np.std((assignment_bboxes[:, 6]), axis=0)),
                                    float((abs(dets[i, 2] - dets[i, 0])) + (abs(dets[i, 3] - dets[i, 1]))),
                                    float(np.min((assignment_bboxes[:, 7]), axis=0)),
                                    float(np.max((assignment_bboxes[:, 7]), axis=0)),
                                    float(np.mean((assignment_bboxes[:, 7]), axis=0)),
                                    float(np.std((assignment_bboxes[:, 7]), axis=0)),
                                    float(((0.5 * (abs(dets[i, 2] - dets[i, 0]))) * (0.5 * (abs(dets[i, 3] - dets[i, 1])))) / ((abs(dets[i, 2] - dets[i, 0])) + (abs(dets[i, 3] - dets[i, 1])))),
                                    float(np.min((assignment_bboxes[:, 8]), axis=0)),
                                    float(np.max((assignment_bboxes[:, 8]), axis=0)),
                                    float(np.mean((assignment_bboxes[:, 8]), axis=0)),
                                    float(np.std((assignment_bboxes[:, 8]), axis=0)),
                                    float(np.min((assignment_bboxes[:, 4]), axis=0)),
                                    float(np.max((assignment_bboxes[:, 4]), axis=0)),
                                    float(np.mean((assignment_bboxes[:, 4]), axis=0)),
                                    float(np.std((assignment_bboxes[:, 4]), axis=0)), float(minimum), float(maximum),
                                    float(max(0, np.mean(iou[iou > 0]))), float(max(0, np.std(iou[iou > 0])))]

    if cal_iou == True:
        if gt_bboxes_img != []:   
            ious_gt = bbox_overlaps(torch.FloatTensor(dets[:, :-1]),
                            torch.FloatTensor(gt_bboxes_img), 
                            mode='iou')
            
            ious_gt = np.asarray(ious_gt)
            gt_labels_img = np.asarray(gt_labels_img)
            ious_gt = [np.where(labels[0][idx] == gt_labels_img, i, 0) for idx, i in enumerate(ious_gt)]
            ious_gt_max = [np.max(x) for x in ious_gt]
            metrics_single_image = np.append(metrics_single_image, np.asmatrix(ious_gt_max).T, axis=1)
        else:
            metrics_single_image = np.append(metrics_single_image, np.zeros((metrics_single_image.shape[0],1)), axis=1)

    return metrics_single_image


def meta_detect_query(model,
                  query_dataloader,
                  val_dataloader,
                  model_cfg,
                  nms_cls_score_thr=0.05,
                  pre_nms_topk=1000,
                  post_nms_topk=150,
                  image_aggregation=torch.sum,
                  img_ids=[],
                  query_size=100,
                  query_score_thr=0.1,
                  config=None,
                  base_target_path=None,
                  step=None,
                  val_dict=None,
                  query_dict=None):
    output, metas = single_gpu_tensor_outputs(MMDataParallel(model, device_ids=[0]),
                                        query_dataloader,
                                        )

    val_output, val_metas = single_gpu_tensor_outputs(MMDataParallel(model, device_ids=[0]),
                                        val_dataloader,
                                        )

    meta_detect, output_metrics = compute_topk_meta_detect_samples(model,
                                                   output,
                                                   metas,
                                                   val_output,
                                                   val_metas,
                                                   model_cfg,
                                                   nms_cls_score_thr,
                                                   pre_nms_topk,
                                                   post_nms_topk,
                                                   n_classes=len(query_dataloader.dataset.CLASSES),
                                                   query_score_thr=query_score_thr,
                                                   base_target_path=base_target_path,
                                                   step=step,
                                                   val_dict=val_dict,
                                                   query_dict=query_dict)
    class_weights = compute_class_weights(config, meta_detect)

    labels = list(output_metrics["labels"].values())

    weighted_meta_detect_scores = [meta_detect[i].squeeze()*class_weights[labels[i]]
                          for i in range(len(meta_detect))]
    query_score = [image_aggregation(e).item() for e in weighted_meta_detect_scores]
    
    query_scores, query_score_ids = torch.topk(
        torch.tensor(query_score), k=query_size)
    query_img_ids = [img_ids[int(i)]
                     for i in query_score_ids.tolist()]
    
    output_metrics.update({"meta_detect": [e.tolist() for e in meta_detect],
                           "query_scores": query_scores.tolist(),
                           "queried_ids": query_img_ids,
                           "class_weights": class_weights.tolist(),
                           "weighted_meta_detect": [e.tolist() for e in weighted_meta_detect_scores]})

    return query_img_ids, output_metrics

def compute_topk_meta_detect_samples(model, output, metas, val_output, val_metas, model_cfg, cls_score_thr, pre_topk=1000, post_topk=150, n_classes=80, query_score_thr=0.0, base_target_path=None, step=None, val_dict=None, query_dict=None):
    image_wise_md_score = []
    detection_dict = {}
    label_dict = {}
    top_boxes_post_nms_dict = {}
    top_probas_post_nms_dict = {}
    top_scores_post_nms_dict = {}
    val_dict = json.load(open(val_dict, "r"))
    print('\nCompute Metrics for Validation Data...')
    for img_id, results in enumerate(tqdm(val_output)):
        image_name = (val_metas[img_id]["img_metas"][0]._data)[0][0]['filename']
        image_id = [int(x['id']) for x in val_dict["images"] if x['file_name'] == image_name.split('/')[-1]]
        gt_bboxes_img = np.asmatrix(np.asarray([x["bbox"] for x in val_dict["annotations"] if int(x["image_id"])==image_id[0]]))
        try:
            gt_bboxes_img[:, 2] += gt_bboxes_img[:, 0]
            gt_bboxes_img[:, 3] += gt_bboxes_img[:, 1]
        except:
            gt_bboxes_img = []
        gt_labels_img = np.array([int(x["category_id"])-1 for x in val_dict["annotations"] if x["image_id"]==image_id[0]])
        results = results[0]
        boxes = results[0]
        probas = results[1]
        scores = results[2]
        if scores is None:
            scores = torch.max(probas, dim=2)[0]
            score_factors = torch.ones_like(scores)
            cls_score_thr = query_score_thr
        elif isinstance(model, TwoStageDetector):
            # num_classes = model_cfg.roi_head.bbox_head.num_classes
            score_factors = torch.ones_like(scores)
            cls_score_thr = query_score_thr
        else:
            score_factors = scores

        top_scores, score_inds = torch.topk(scores.squeeze(),
                                            pre_topk
                                            )

        score_inds = score_inds[top_scores >= query_score_thr]
        
        top_scores = scores[:, score_inds]
        score_factors = score_factors[:, score_inds]
        top_boxes = boxes[:, score_inds, :]
        top_probas = probas[:, score_inds, :]
        if isinstance(model, TwoStageDetector) and len(score_inds) > 0:
            top_probas = top_scores.unsqueeze(2) * \
                torch.nn.functional.one_hot(
                    top_probas[:, :, -1].squeeze().long(), num_classes=n_classes+1).unsqueeze(0)
        

        if len(score_inds) > 0:
            nms_cfg = model_cfg.test_cfg.nms if "nms" in model_cfg.test_cfg else model_cfg.test_cfg.rcnn.nms
            dets, labels, inds = multiclass_nms(top_boxes.squeeze(0),  # results[0].squeeze(),
                                                # results[1].squeeze(),
                                                top_probas.squeeze(0),
                                                cls_score_thr,
                                                nms_cfg,
                                                max_num=post_topk,
                                                # results[2].squeeze(),
                                                score_factors=score_factors,
                                                return_inds=True
                                                )
            metrics_single_image = metric_calculation(np.asmatrix(dets.cpu()),
                               np.asmatrix(labels.cpu()),
                               np.asmatrix(top_boxes.squeeze(0).cpu()),
                               np.asmatrix(top_probas.squeeze(0).cpu()),
                               nms_cfg,
                               score_factors=score_factors,
                               query_score_thr=query_score_thr,
                               gt_bboxes_img=gt_bboxes_img,
                               gt_labels_img=gt_labels_img,
                               cal_iou=True
                               )

            top_probas = probas[:, score_inds, :]
            inds = torch.div(inds, n_classes,
                             rounding_mode="floor").long()
            # i = inds[0]
            # idx = i // 80
            post_probs = top_probas[:, inds, :-1]

            labels_temp = [x for x in np.array(labels.cpu())]
            metrics_single_image = np.concatenate((metrics_single_image, np.asmatrix(np.array(dets.cpu())), np.asmatrix(labels_temp).T, np.asmatrix(np.array(post_probs.cpu()))), axis=1)
            header_str = META_DETECT_METRICS + ['true_iou'] + ['xmin', 'ymin', 'xmax', 'ymax', 's', 'category_idx'] + [f"prob_{i}" for i in range(n_classes)]
            metrics_single_image = pd.DataFrame(metrics_single_image, columns=header_str)
            metrics_single_image['file_path'] = image_name

            try:
                metrics_matrix = metrics_matrix.append(metrics_single_image, ignore_index = True)
            except:
                metrics_matrix = metrics_single_image

    header_str = META_DETECT_METRICS + ['true_iou'] + ['xmin', 'ymin', 'xmax', 'ymax', 's', 'category_idx'] + [f"prob_{i}" for i in range(n_classes)] + ['file_path']
    target_path = osp.join(base_target_path, f"step_{step}/val_metrics.csv")
    # pd.DataFrame(metrics_matrix).to_csv("/home/schubert/file_score=" + str(score_threshold) + ".csv", header=header_str)
    md_metrics_val = pd.DataFrame(metrics_matrix)
    md_metrics_val.to_csv(target_path, header=header_str)

    # meta_model = xgb.XGBClassifier(tree_method="gpu_hist", gpu_id=0, use_label_encoder=False, max_depth=3, n_estimators=29, reg_alpha=1.5, reg_lambda=0.0, learning_rate=0.3)
    meta_model = xgb.XGBClassifier(tree_method="hist", use_label_encoder=False, max_depth=3, n_estimators=29, reg_alpha=1.5, reg_lambda=0.0, learning_rate=0.3)
    meta_model.fit(md_metrics_val.drop(['true_iou', 'file_path'],axis=1), md_metrics_val['true_iou'].round(0))
    
    print('\nCompute Metrics for Unlabeled Data...')
    query_dict = json.load(open(query_dict, "r"))
    for img_id, results in enumerate(tqdm(output)):
        image_name = (metas[img_id]["img_metas"][0]._data)[0][0]['filename']
        image_id = [int(x['id']) for x in query_dict["images"] if x['file_name'] == image_name.split('/')[-1]]
        gt_bboxes_img = np.asmatrix(np.asarray([x["bbox"] for x in query_dict["annotations"] if int(x["image_id"])==image_id[0]]))
        try:
            gt_bboxes_img[:, 2] += gt_bboxes_img[:, 0]
            gt_bboxes_img[:, 3] += gt_bboxes_img[:, 1]
        except:
            gt_bboxes_img = []
        gt_labels_img = np.array([int(x["category_id"])-1 for x in query_dict["annotations"] if x["image_id"]==image_id[0]])
        results = results[0]
        boxes = results[0]
        probas = results[1]
        scores = results[2]
        if scores is None:
            scores = torch.max(probas, dim=2)[0]
            score_factors = torch.ones_like(scores)
            cls_score_thr = query_score_thr
        elif isinstance(model, TwoStageDetector):
            # num_classes = model_cfg.roi_head.bbox_head.num_classes
            score_factors = torch.ones_like(scores)
            cls_score_thr = query_score_thr
        else:
            score_factors = scores

        top_scores, score_inds = torch.topk(scores.squeeze(),
                                            pre_topk
                                            )
        
        score_inds = score_inds[top_scores >= query_score_thr]
        
        top_scores = scores[:, score_inds]
        score_factors = score_factors[:, score_inds]
        top_boxes = boxes[:, score_inds, :]
        top_probas = probas[:, score_inds, :]
        if isinstance(model, TwoStageDetector) and len(score_inds) > 0:
            top_probas = top_scores.unsqueeze(2) * \
                torch.nn.functional.one_hot(
                    top_probas[:, :, -1].squeeze().long(), num_classes=n_classes+1).unsqueeze(0)
        

        if len(score_inds) > 0:
            nms_cfg = model_cfg.test_cfg.nms if "nms" in model_cfg.test_cfg else model_cfg.test_cfg.rcnn.nms
            dets, labels, inds = multiclass_nms(top_boxes.squeeze(0),  # results[0].squeeze(),
                                                # results[1].squeeze(),
                                                top_probas.squeeze(0),
                                                cls_score_thr,
                                                nms_cfg,
                                                max_num=post_topk,
                                                # results[2].squeeze(),
                                                score_factors=score_factors,
                                                return_inds=True
                                                )
            metrics_single_image = metric_calculation(np.asmatrix(dets.cpu()),
                               np.asmatrix(labels.cpu()),
                               np.asmatrix(top_boxes.squeeze(0).cpu()),
                               np.asmatrix(top_probas.squeeze(0).cpu()),
                               nms_cfg,
                               score_factors=score_factors,
                               query_score_thr=query_score_thr,
                               gt_bboxes_img=gt_bboxes_img,
                               gt_labels_img=gt_labels_img,
                               cal_iou=True
                               )

            top_probas = probas[:, score_inds, :]
            inds = torch.div(inds, n_classes,
                             rounding_mode="floor").long()
            # i = inds[0]
            # idx = i // 80
            post_probs = top_probas[:, inds, :-1]

            labels_temp = [x for x in np.array(labels.cpu())]
            metrics_single_image = np.concatenate((metrics_single_image, np.asmatrix(np.array(dets.cpu())), np.asmatrix(labels_temp).T, np.asmatrix(np.array(post_probs.cpu()))), axis=1)
            header_str = META_DETECT_METRICS + ['true_iou'] + ['xmin', 'ymin', 'xmax', 'ymax', 's', 'category_idx'] + [f"prob_{i}" for i in range(n_classes)]
            metrics_single_image = pd.DataFrame(metrics_single_image, columns=header_str)
            metrics_single_image['file_path'] = image_name

            pred_tp = meta_model.predict_proba(metrics_single_image.drop(['file_path', 'true_iou'],axis=1))[:,1]
            metrics_single_image['pred_iou'] = pred_tp
            uncertainty = - pred_tp * np.log2(pred_tp) - (1-pred_tp) * (1-np.log(pred_tp))
            metrics_single_image['uncertainty'] = uncertainty

            try:
                metrics_matrix_unlabeled = metrics_matrix_unlabeled.append(metrics_single_image, ignore_index = True)
            except:
                metrics_matrix_unlabeled = metrics_single_image
            
            # entropy = - torch.sum(post_probs * torch.log(post_probs), dim=2)

            # image_wise_md_score.append(torch.tensor([pred_tp]) * entropy)
            image_wise_md_score.append(torch.tensor([pred_tp]))
            detection_dict[img_id] = dets.tolist()
            label_dict[img_id] = labels.tolist()
            top_boxes_post_nms_dict[img_id] = top_boxes[:, inds, :].tolist()
            top_probas_post_nms_dict[img_id] = post_probs.tolist()
            top_scores_post_nms_dict[img_id] = top_scores[:, inds].tolist()
        
        else:
            image_wise_md_score.append(torch.tensor([[0.0]]))
            detection_dict[img_id] = []
            label_dict[img_id] = []
            top_boxes_post_nms_dict[img_id] = []
            top_probas_post_nms_dict[img_id] = []
            top_scores_post_nms_dict[img_id] = []
            

    header_str = META_DETECT_METRICS + ['true_iou'] + ['xmin', 'ymin', 'xmax', 'ymax', 's', 'category_idx'] + [f"prob_{i}" for i in range(n_classes)] + ['file_path', 'pred_iou', 'uncertainty']
    target_path = osp.join(base_target_path, f"step_{step}/unlabeled_metrics.csv")
    metrics_matrix_unlabeled.to_csv(target_path, header=header_str)

    return image_wise_md_score, {"detections": detection_dict,
                                "labels": label_dict,
                                "top_boxes_post_nms": top_boxes_post_nms_dict,
                                "top_probas_post_nms": top_probas_post_nms_dict,
                                "top_scores_post_nms": top_scores_post_nms_dict,
                                }