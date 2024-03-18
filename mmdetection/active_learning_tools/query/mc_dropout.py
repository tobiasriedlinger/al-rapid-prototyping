import time
import torch

from mmcv.parallel import MMDataParallel

from mmdet.apis.test import single_gpu_tensor_outputs
from mmdet.models import TwoStageDetector
from mmdet.core.post_processing.bbox_nms import multiclass_nms
from active_learning_tools.query.class_weights import compute_class_weights


def dropout_query(model,
                  query_dataloader,
                  model_cfg,
                  nms_cls_score_thr=0.05,
                  pre_nms_topk=1000,
                  post_nms_topk=150,
                  image_aggregation=torch.sum,
                  img_ids=[],
                  query_size=100,
                  query_score_thr=0.1,
                  config=None,
                  timing_dict={}):
    tic = time.time()
    output, metas = single_gpu_tensor_outputs(MMDataParallel(model, device_ids=[0]),
                                query_dataloader,
                                dropout=config.splits.query.n_samples
                                )
            
    timing_dict.update({"query_inference": float(time.time() - tic)})
    
    tic = time.time()
    dropout_score, output_metrics = compute_topk_dropout_samples(model,
                                                                 output,
                                                   model_cfg,
                                                   nms_cls_score_thr,
                                                   pre_nms_topk,
                                                   post_nms_topk,
                                                   n_classes=len(query_dataloader.dataset.CLASSES),
                                                   query_score_thr=query_score_thr)
    
    class_weights = compute_class_weights(config, dropout_score)
    labels = list(output_metrics["labels"].values())
    weighted_dropout_scores = [dropout_score[i].squeeze()*class_weights[labels[i]]
                          for i in range(len(dropout_score))]
    query_score = [image_aggregation(e).item() if e.numel() > 0 else 0 for e in weighted_dropout_scores]
    query_scores, query_score_ids = torch.topk(
        torch.tensor(query_score), k=query_size)
    query_img_ids = [img_ids[int(i)]
                     for i in query_score_ids.tolist()]
    output_metrics.update({"dropout_score": [e.tolist() for e in dropout_score],
                           "query_scores": query_scores.tolist(),
                           "queried_ids": query_img_ids,
                           "class_weights": class_weights.tolist(),
                           "weighted_dropout_score": [e.tolist() for e in weighted_dropout_scores]})
    timing_dict.update({"query_construction": float(time.time() - tic)})

    return query_img_ids, output_metrics, timing_dict

def compute_topk_dropout_samples(model, 
                                 output, 
                                 model_cfg, 
                                 cls_score_thr, 
                                 pre_topk=1000, 
                                 post_topk=150, 
                                 n_classes=80, 
                                 query_score_thr=0.0
                                 ):
    image_wise_mc_score = []
    detection_dict = {}
    label_dict = {}
    top_boxes_post_nms_dict = {}
    top_probas_post_nms_dict = {}
    top_scores_post_nms_dict = {}
    
    min_num_predictions = min([len(d["means"]["bboxes"]) for d in output])
    box_mc_stds = torch.stack([d["stds"]["bboxes"][:min_num_predictions] for d in output], dim=0)
    proba_mc_stds = torch.stack([d["stds"]["probas"][:min_num_predictions] for d in output], dim=0)
    score_mc_stds = torch.stack([d["stds"]["scores"][:min_num_predictions] for d in output], dim=0)
    box_mc_means = torch.stack([d["means"]["bboxes"][:min_num_predictions] for d in output], dim=0)
    proba_mc_means = torch.stack([d["means"]["probas"][:min_num_predictions] for d in output], dim=0)
    score_mc_means = torch.stack([d["means"]["scores"][:min_num_predictions] for d in output], dim=0)
    
    # [Batch, Anchor, Feature]
    box_std_mean = torch.mean(box_mc_stds, dim=0, keepdim=True)
    box_std_std = torch.std(box_mc_stds, dim=0, keepdim=True) + 1e-9
    proba_std_mean = torch.mean(proba_mc_stds, dim=0, keepdim=True)
    proba_std_std = torch.std(proba_mc_stds, dim=0, keepdim=True) + 1e-9
    score_std_mean = torch.mean(score_mc_stds, dim=0, keepdim=True)
    score_std_std = torch.std(score_mc_stds, dim=0, keepdim=True) + 1e-9
    
    
    id_range = len(output)
    del output

    for img_id in range(id_range):
        boxes = box_mc_means[img_id, ...]
        probas = proba_mc_means[img_id, ...]
        scores = score_mc_means[img_id, ...]
        

        top_scores, score_inds = torch.topk(scores.squeeze(),
                                            pre_topk
                                            )
        score_inds = score_inds[top_scores >= query_score_thr]
        top_scores = scores[score_inds, :].unsqueeze(0)
        score_factors = top_scores
        top_boxes = boxes[score_inds, :].unsqueeze(0)
        top_probas = probas[score_inds, :].unsqueeze(0)
        if isinstance(model, TwoStageDetector) and len(score_inds) > 0:
                top_probas = top_scores * \
                    torch.nn.functional.one_hot(
                        top_probas[:, :, -1].squeeze().long(), num_classes=n_classes+1).unsqueeze(0)

        if len(score_inds) > 0:
            nms_cfg = model_cfg.test_cfg.nms if "nms" in model_cfg.test_cfg else model_cfg.test_cfg.rcnn.nms
            dets, labels, inds = multiclass_nms(top_boxes.squeeze(0),
                                                top_probas.squeeze(0),
                                                cls_score_thr,
                                                nms_cfg,
                                                max_num=post_topk,
                                                score_factors=score_factors,
                                                return_inds=True
                                                )
            top_probas = probas[score_inds, :].unsqueeze(0)

            inds = torch.div(inds, n_classes,
                             rounding_mode="floor").long()
            post_probs = top_probas[:, inds, :-1]
            
            boxes_std = box_mc_stds[img_id, score_inds, :].unsqueeze(0)
            box_std_std_temp = box_std_std[:, score_inds, :]
            box_std_mean_temp = box_std_mean[:, score_inds, :]
            
            proba_std_std_temp = proba_std_std[:, score_inds, :]
            proba_std_mean_temp = proba_std_mean[:, score_inds, :]
            
            score_std = score_mc_stds[img_id, score_inds].unsqueeze(0)
            score_std_std_temp = score_std_std[:, score_inds]
            score_std_mean_temp = score_std_mean[:, score_inds]
            
            boxes_std_norm = (boxes_std[:, inds, :] - box_std_mean_temp[:, inds, :]) / box_std_std_temp[:, inds, :]
            probas_std_norm = (post_probs - proba_std_mean_temp[:, inds, :-1]) / proba_std_std_temp[:, inds, :-1]
            score_std_norm = (score_std[:, inds] - score_std_mean_temp[:, inds]) / score_std_std_temp[:, inds]
            
            uncertainties = torch.cat([boxes_std_norm, probas_std_norm, score_std_norm], dim=2)

            if torch.prod(torch.tensor(inds.shape)) > 0:
                max_uncertainty = torch.max(uncertainties, dim=2)[0]
            else:
                max_uncertainty = torch.tensor([[0.0]])
            
            image_wise_mc_score.append(max_uncertainty)
            detection_dict[img_id] = dets.tolist()
            label_dict[img_id] = labels.tolist()
            top_boxes_post_nms_dict[img_id] = top_boxes[:, inds, :].tolist()
            top_probas_post_nms_dict[img_id] = post_probs.tolist()
            top_scores_post_nms_dict[img_id] = scores[inds, :].tolist()
        else:
            image_wise_mc_score.append(torch.tensor([[0.0]]))
            detection_dict[img_id] = []
            label_dict[img_id] = []
            top_boxes_post_nms_dict[img_id] = []
            top_probas_post_nms_dict[img_id] = []
            top_scores_post_nms_dict[img_id] = []

    return image_wise_mc_score, {"detections": detection_dict,
                                "labels": label_dict,
                                "top_boxes_post_nms": top_boxes_post_nms_dict,
                                "top_probas_post_nms": top_probas_post_nms_dict,
                                "top_scores_post_nms": top_scores_post_nms_dict,
                                }