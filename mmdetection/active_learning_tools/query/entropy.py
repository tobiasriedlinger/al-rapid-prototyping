import time
import torch

from mmcv.parallel import MMDataParallel

from mmdet.apis.test import single_gpu_tensor_outputs
from mmdet.models import TwoStageDetector
from mmdet.core.post_processing.bbox_nms import multiclass_nms
from active_learning_tools.query.class_weights import compute_class_weights


def entropy_query(model, 
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
                                        )
    timing_dict.update({"query_inference": float(time.time() - tic)})
    
    tic = time.time()    
    entropy, output_metrics = compute_topk_entropy(model,
                                                   output,
                                                   model_cfg,
                                                   nms_cls_score_thr,
                                                   pre_nms_topk,
                                                   post_nms_topk,
                                                   n_classes=len(query_dataloader.dataset.CLASSES),
                                                   query_score_thr=query_score_thr)
    class_weights = compute_class_weights(config, entropy)
    labels = list(output_metrics["labels"].values())
    weighted_entropies = [entropy[i].squeeze()*class_weights[labels[i]]
                          for i in range(len(entropy))]
    query_score = [image_aggregation(e).item() if e.numel() > 0 else 0 for e in weighted_entropies]
    query_scores, query_score_ids = torch.topk(
        torch.tensor(query_score), k=query_size)
    query_img_ids = [img_ids[int(i)]
                     for i in query_score_ids.tolist()]
    output_metrics.update({"entropies": [e.tolist() for e in entropy],
                           "query_scores": query_scores.tolist(),
                           "queried_ids": query_img_ids,
                           "class_weights": class_weights.tolist(),
                           "weighted_entropies": [e.tolist() for e in weighted_entropies]})
    timing_dict.update({"query_construction": float(time.time() - tic)})

    return query_img_ids, output_metrics, timing_dict


def compute_topk_entropy(model, output, model_cfg, cls_score_thr, pre_topk=1000, post_topk=150, n_classes=80, query_score_thr=0.0):
    image_wise_entropy = []
    detection_dict = {}
    label_dict = {}
    top_boxes_post_nms_dict = {}
    top_probas_post_nms_dict = {}
    top_scores_post_nms_dict = {}
    for img_id, results in enumerate(output):
        results = results[0]
        boxes = results[0]
        probas = results[1]
        scores = results[2]
        if scores is None:
            scores = torch.max(probas, dim=2)[0]
            score_factors = torch.ones_like(scores)
            cls_score_thr = query_score_thr
        elif isinstance(model, TwoStageDetector):
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
            dets, labels, inds = multiclass_nms(top_boxes.squeeze(0),
                                                top_probas.squeeze(0),
                                                cls_score_thr,
                                                nms_cfg,
                                                max_num=post_topk,
                                                score_factors=score_factors,
                                                return_inds=True
                                                )
            top_probas = probas[:, score_inds, :]
            inds = torch.div(inds, n_classes,
                             rounding_mode="floor").long()
            post_probs = top_probas[:, inds, :-1]
            post_probs = post_probs / torch.sum(post_probs, dim=2, keepdim=True)
            if torch.prod(torch.tensor(inds.shape)) > 0:
                entropy = - torch.sum(post_probs * torch.log(post_probs), dim=2)
            else:
                entropy = torch.tensor([[0.0]])
            image_wise_entropy.append(entropy)
            detection_dict[img_id] = dets.tolist()
            label_dict[img_id] = labels.tolist()
            top_boxes_post_nms_dict[img_id] = top_boxes[:, inds, :].tolist()
            top_probas_post_nms_dict[img_id] = post_probs.tolist()
            top_scores_post_nms_dict[img_id] = top_scores[:, inds].tolist()
        else:
            image_wise_entropy.append(torch.tensor([[0.0]]))
            detection_dict[img_id] = []
            label_dict[img_id] = []
            top_boxes_post_nms_dict[img_id] = []
            top_probas_post_nms_dict[img_id] = []
            top_scores_post_nms_dict[img_id] = []

    return image_wise_entropy, {"detections": detection_dict,
                                "labels": label_dict,
                                "top_boxes_post_nms": top_boxes_post_nms_dict,
                                "top_probas_post_nms": top_probas_post_nms_dict,
                                "top_scores_post_nms": top_scores_post_nms_dict,
                                }
