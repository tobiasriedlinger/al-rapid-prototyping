import cv2
import numpy as np


def extract_segments(img_ls):
    segments = []
    for img in img_ls:
        img = (img.numpy() < 1.)
        mask = img.astype("uint8").transpose(1, 2, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        segments.append(filter_segments(contours))
    return segments
            
def is_contained_in(c1, c2):
    min_c1 = np.squeeze(c1).min(axis=0)
    max_c1 = np.squeeze(c1).max(axis=0)
    min_c2 = np.squeeze(c2).min(axis=0)
    max_c2 = np.squeeze(c2).max(axis=0)
    
    return np.all(np.logical_and(min_c1 > min_c2, max_c1 < max_c2))

def insert_into(c1, c2):
    c2_last = np.expand_dims(c2[-1], axis=0)
    closest_id = np.argmin((np.abs(c1 - c2_last)**2).sum(axis=1))
    closest_c1_node = np.expand_dims(c1[closest_id], axis=0)
    
    return np.concatenate([c2_last, c1[closest_id:], c1[:closest_id], closest_c1_node, c2_last, c2])
    

def filter_segments(contours):
    if len(contours) == 1:
        return [np.squeeze(contours).reshape(-1).tolist()]
    segments = []
    while len(contours) > 0:
        c = contours.pop(0)
        if len(contours) == 0:
            segments.append(c.reshape(-1).tolist())
        else:
            c = np.squeeze(c)
            min_c = c.min(axis=0)
            max_c = c.max(axis=0)
            for i, c_other in enumerate(contours):
                c_other = np.squeeze(c_other)
                if is_contained_in(c, c_other):
                    contours.pop(i)
                    contours.append(insert_into(c, c_other))
                elif is_contained_in(c_other, c):
                    contours.pop(i)
                    contours.append(insert_into(c_other, c))
                
                else:
                    area = np.prod(max_c - min_c)
                    if area > 25:
                        segments.append(c.reshape(-1).tolist())
                        continue
        
    return segments
