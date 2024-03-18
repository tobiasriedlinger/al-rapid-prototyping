"""
Mutual information query function.
Computes the mutual information of MC dropout mean entropy and mean of individual entropies.
"""
import time
import torch

from mmcv.parallel import MMDataParallel

from mmdet.apis.test import single_gpu_tensor_outputs
from mmdet.models import TwoStageDetector
from mmdet.core.post_processing.bbox_nms import multiclass_nms
from active_learning_tools.query.class_weights import compute_class_weights


def mutual_information_query(model,
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
    """Uses model to perform mutual information query on a query dataloader.
    Returns image ids from dataset and query information.

    Args:
        model (BaseDetector): Model to perform query with.
        query_dataloader (torch.DataLoader): Dataloader with query dataset.
        model_cfg (Config): Configuration object of model.
        nms_cls_score_thr (float, optional): Class score threshold for NMS. Defaults to 0.05.
        pre_nms_topk (int, optional): Number of top k boxes pre NMS. Defaults to 1000.
        post_nms_topk (int, optional): Number of top k boxes post NMS. Defaults to 150.
        image_aggregation (str, optional): Aggregation function to compute image query score from predictions query scores. Defaults to torch.sum.
        img_ids (list, optional): Image ids of query data w.r.t. original dataset. Defaults to [].
        query_size (int, optional): Number of images to query from query_dataloader. Defaults to 100.
        query_score_thr (float, optional): Confidence score threshold to use on boxes in order to count to the query prediction. Defaults to 0.1.
        config (Config, optional): Config object with query information. Defaults to None.

    Returns:
        tuple(list, dict): Queried image ids w.r.t. original dataset, query information dictionary including query scores.
    """
    tic = time.time()
    output, _ = single_gpu_tensor_outputs(MMDataParallel(model, device_ids=[0]),
                                        query_dataloader,
                                        dropout=config.splits.query.n_samples
                                        )
    timing_dict.update({"query_inference": float(time.time() - tic)})

    tic = time.time()
    mutual_information, output_metrics = compute_topk_mutual_information_samples(model,
                                                   output,
                                                   model_cfg,
                                                   nms_cls_score_thr,
                                                   pre_nms_topk,
                                                   post_nms_topk,
                                                   n_classes=len(query_dataloader.dataset.CLASSES),
                                                   query_score_thr=query_score_thr)
    
    class_weights = compute_class_weights(config, mutual_information)

    # Predicted labels for class weighting
    labels = list(output_metrics["labels"].values())
    
    weighted_mutual_information_scores = [mutual_information[i].squeeze()*class_weights[labels[i]]
                          for i in range(len(mutual_information))]
    query_score = [image_aggregation(e).item() if e.numel() > 0 else 0 for e in weighted_mutual_information_scores]
    query_scores, query_score_ids = torch.topk(torch.tensor(query_score), k=query_size)

    # top-k-indices to dataset indices
    query_img_ids = [img_ids[int(i)] for i in query_score_ids.tolist()]
    output_metrics.update({"mutual_information": [e.tolist() for e in mutual_information],
                           "query_scores": query_scores.tolist(),
                           "queried_ids": query_img_ids,
                           "class_weights": class_weights.tolist(),
                           "weighted_mutual_information": [e.tolist() for e in weighted_mutual_information_scores]})
    timing_dict.update({"query_construction": float(time.time() - tic)})

    return query_img_ids, output_metrics, timing_dict

def compute_topk_mutual_information_samples(model, 
                                            output, 
                                            model_cfg, 
                                            cls_score_thr, 
                                            pre_topk=1000, 
                                            post_topk=150, 
                                            n_classes=80, 
                                            query_score_thr=0.0
                                            ):
    """Given dropout model output, computes mutual information scores for top k
    predictions.

    Args:
        model (BaseDetector): Model to determine architecture type.
        output (list[dict]): Tensor output samples from model.
        model_cfg (Config): Configuration object of model.
        cls_score_thr (float): Class score threshold for NMS.
        pre_topk (int, optional): Number of top k boxes pre NMS.. Defaults to 1000.
        post_topk (int, optional): Number of top k boxes post NMS.. Defaults to 150.
        n_classes (int, optional): Number of predicted categories. Defaults to 80.
        query_score_thr (float, optional): Confidence score threshold to use on boxes in order to count to the query prediction. Defaults to 0.0.

    Returns:
        tiple(list, dict): Mutual information query scores per image and query information dict with keys ["detections", "labels", "top_boxes_post_nms", "top_probas_post_nms", "top_scores_post_nms"]
    """
    image_wise_mi_score = []
    detection_dict = {}
    label_dict = {}
    top_boxes_post_nms_dict = {}
    top_probas_post_nms_dict = {}
    top_scores_post_nms_dict = {}

    # Compute MC dropout means
    min_num_predictions = min([len(d["means"]["bboxes"]) for d in output])
    box_mc_means = torch.stack([d["means"]["bboxes"][:min_num_predictions] for d in output], dim=0)
    proba_mc_means = torch.stack([d["means"]["probas"][:min_num_predictions] for d in output], dim=0)
    score_mc_means = torch.stack([d["means"]["scores"][:min_num_predictions] for d in output], dim=0)
    entropies = torch.stack([d["entropy"][:, :min_num_predictions] for d in output], dim=0)

    for img_id, results in enumerate(output):
        # probas_orig = torch.cat([t[1] for t in results["samples"][0]], dim=0)
        boxes = box_mc_means[img_id, ...]
        probas = proba_mc_means[img_id, ...]
        scores = score_mc_means[img_id, ...]
        mean_entropy = entropies[img_id, ...]

        # Filter for top k predicitons
        top_scores, score_inds = torch.topk(scores.squeeze(),
                                            pre_topk
                                            )
        score_inds = score_inds[top_scores >= query_score_thr]
        top_scores = scores[score_inds, :].unsqueeze(0)
        score_factors = top_scores
        top_boxes = boxes[score_inds, :].unsqueeze(0)
        top_probas = probas[score_inds, :].unsqueeze(0)
        top_entropies = mean_entropy[:, score_inds]

        # Distribute class-wise scores over one-hot vectors (due to class-wise bounding box regressions)
        if isinstance(model, TwoStageDetector) and len(score_inds) > 0:
                top_probas = top_scores * \
                    torch.nn.functional.one_hot(
                        top_probas[:, :, -1].squeeze().long(), num_classes=n_classes+1).unsqueeze(0)

        if len(score_inds) > 0:
            nms_cfg = model_cfg.test_cfg.nms if "nms" in model_cfg.test_cfg else model_cfg.test_cfg.rcnn.nms
            # Perform NMS
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

            # Filter for NMS survivors

            single_entropy = top_entropies[:, inds]
            post_probs = top_probas[:, inds, :-1]
            post_probs = post_probs / torch.sum(post_probs, dim=2, keepdim=True)

            # MC mean entropy
            if torch.prod(torch.tensor(inds.shape)) > 0:
                mean_entropy = - torch.sum(post_probs * torch.log(post_probs), dim=2).squeeze()
                # Mean entropy of dropout samples
                mean_dropout_entropy = torch.mean( single_entropy, dim=0)
            else:
                mean_entropy = torch.tensor([[0.0]])
                mean_dropout_entropy = torch.tensor([[0.0]])
            
            image_wise_mi_score.append(abs(mean_entropy - mean_dropout_entropy))
            detection_dict[img_id] = dets.tolist()
            label_dict[img_id] = labels.tolist()
            top_boxes_post_nms_dict[img_id] = top_boxes[:, inds, :].tolist()
            top_probas_post_nms_dict[img_id] = post_probs.tolist()
            top_scores_post_nms_dict[img_id] = scores[inds, :].tolist()
        else:
            image_wise_mi_score.append(torch.tensor([[0.0]]))
            detection_dict[img_id] = []
            label_dict[img_id] = []
            top_boxes_post_nms_dict[img_id] = []
            top_probas_post_nms_dict[img_id] = []
            top_scores_post_nms_dict[img_id] = []

    return image_wise_mi_score, {"detections": detection_dict,
                                "labels": label_dict,
                                "top_boxes_post_nms": top_boxes_post_nms_dict,
                                "top_probas_post_nms": top_probas_post_nms_dict,
                                "top_scores_post_nms": top_scores_post_nms_dict,
                                }