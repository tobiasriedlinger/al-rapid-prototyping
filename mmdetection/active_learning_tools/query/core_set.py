import time
import os.path as osp
import copy
from mmdet.apis.test import single_gpu_tensor_outputs
import torch

from mmdet.datasets import build_dataset, build_dataloader
from mmcv.parallel import MMDataParallel
from mmdet.models import TwoStageDetector

def cosine_distance(u, v):
    """

    Args:
        u ([type]): [k, h, w, f]
        v ([type]): [n, h, w, f]
    """
    u = u.view(u.shape[0], 1, -1)
    v = v.view(1, v.shape[0], -1)
    scalar_prods = u * v # [k, n, h*w*f]
    scalar_prods = scalar_prods.sum(dim=(2)) # [k, n]
    res = (scalar_prods / torch.norm(v, dim=2) / torch.norm(u, dim=2)).unsqueeze(0)
    
    return res

def l2_distance(u, v):
    """[summary]

    Args:
        u ([type]): [1, h, w, f]
        v ([type]): [n, h, w, f]
    """
    # diff = u.expand_as(v) - v
    
    # return torch.norm(diff.view(diff.shape[0], -1), dim=1)
    return torch.cdist(u.view(1, u.shape[0], -1), v.view(1, v.shape[0], -1))
    

distances = {"cosine": cosine_distance,
             "l2": l2_distance}

def combine_features(feats, model):
    if isinstance(model, TwoStageDetector):
        return torch.stack(feats, dim=0)
    else:
        for i, phi in enumerate(feats):
            # print(phi)
            if len(phi) == 1:
                if phi[0][2] is None:
                    feats[i] = torch.cat(phi[0][:2], dim=2)
                else:
                    phi = list(phi[0])
                    phi[2] = phi[2].unsqueeze(-1)
                    feats[i] = torch.cat(phi, dim=2)
            # else:
            #     phi = list(phi)
            #     phi[2] = phi[2].unsqueeze(-1)
            #     feats[i] = torch.cat(phi, dim=2)
            else:
                phi = list(phi)
                if len(phi) > 2:
                    phi[2] = phi[2].unsqueeze(-1)
                    feats[i] = torch.cat(phi, dim=2)
                else:
                    # print(phi)
                # for j, l in enumerate(phi):
                    # phi[j] = torch.cat([t.reshape(1, 1, -1) for t in l], dim=2).unsqueeze(-1)
                    phi = phi[0]
                    phi = torch.cat([t.reshape(1, 1, -1) for t in phi[0]], dim=2).unsqueeze(-1)
                    feats[i] = phi
    
    return torch.cat(feats, dim=0)
    

def core_set_query(model,
                   query_dataloader,
                #    labeled_features,
                #    unlabeled_features,
                   img_ids=[],
                   query_size=100,
                   config=None,
                   annotation_path="",
                   timing_dict={}):
    
    img_id_copy = copy.deepcopy(img_ids)
    train_dict = copy.deepcopy(config.data.train)
    train_dict["ann_file"] = osp.join(
            annotation_path, "labeled_anns.json")
    train_dict["pipeline"] = config.data.test.pipeline
    train_ds = build_dataset(train_dict)
    train_dl = build_dataloader(train_ds,
                                1,
                                4,
                                seed=config.seed,
                                runner_type="EpochBasedRunner")
    config = config.splits.query
    tic = time.time()
    labeled_features, _ = single_gpu_tensor_outputs(MMDataParallel(model, device_ids=[0]),
                                                    train_dl,
                                                    activation=False
                                                    )
    unlabeled_features, _ = single_gpu_tensor_outputs(MMDataParallel(model, device_ids=[0]),
                                                      query_dataloader,
                                                      activation=False)
    timing_dict.update({"query_inference": float(time.time() - tic)})

    tic = time.time()
    
    query_img_ids = []
    # print(labeled_features[0][0][2].shape)
    labeled_features = combine_features(labeled_features, model)
    unlabeled_features = combine_features(unlabeled_features, model)

    # [1, L, U]
    dist = distances[config.distance](labeled_features, unlabeled_features)
    for _ in range(query_size):
        centroid_id = torch.max(torch.min(dist, dim=1)[0], dim=1)[1].squeeze()
        query_img_ids.append(img_id_copy.pop(centroid_id.tolist()))

        dist = torch.cat([dist[:, :, :centroid_id], dist[:, :, centroid_id+1:]], dim=2) # [1, L, U-1]
                
        labeled_features = torch.cat([labeled_features, unlabeled_features[centroid_id, ...].unsqueeze(0)], dim=0)
        unlabeled_features = torch.cat([unlabeled_features[:centroid_id, ...], unlabeled_features[centroid_id+1:, ...]], dim=0)

        dist = torch.cat([dist, distances[config.distance](labeled_features[-1, ...].unsqueeze(0), unlabeled_features)], dim=1)
        # [1, L+1, U-1]
        
        # min_distances = []
        
        # for f in unlabeled_features:
        #     dist = torch.min(distances[config.distance](f.unsqueeze(0), labeled_features))
        #     min_distances.append(dist)
            
        # max_dist, max_dist_id = torch.max(torch.stack(min_distances, dim=0), dim=0)
    timing_dict.update({"query_construction": float(time.time() - tic)})
        
    return query_img_ids, {"queried_ids": query_img_ids}, timing_dict
 