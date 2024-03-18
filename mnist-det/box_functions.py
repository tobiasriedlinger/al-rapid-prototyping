import numpy as np
import torch
import torchvision


def gen_boxes_from_grayscale(img_tensors, img_size):
	boxes = []
	for gsc_img in img_tensors:
		x = torch.amin(gsc_img, 1)
		mask = x < 1.
		n = torch.arange(img_size)
		xmin = torch.min(torch.masked_select(n, mask))
		xmax = torch.max(torch.masked_select(n, mask))
		y = torch.amin(gsc_img, 2)
		mask = y < 1.
		ymin = torch.min(torch.masked_select(n, mask))
		ymax = torch.max(torch.masked_select(n, mask))

		boxes.append(torch.Tensor([xmin, ymin, xmax, ymax]))

	return boxes

def gen_boxes_from_segments(segments):
	boxes = []
	for seg in segments:
		segs = [np.array(s).reshape(-1, 2) for s in seg]
		mins = [np.min(s, axis=0) for s in segs]
		maxs = [np.max(s, axis=0) for s in segs]
		if len(mins) > 1:
			mins = np.min(np.stack(mins, axis=0), axis=0)
			maxs = np.max(np.stack(maxs, axis=0), axis=0)
		else:
			mins = mins[0]
			maxs = maxs[0]
		box = mins.tolist() + maxs.tolist()
		boxes.append(torch.tensor(box))
	
	return boxes

def boxes_too_close(boxes):
	boxes = torch.stack(boxes)
	inter, union = torchvision.ops.boxes._box_inter_union(boxes, boxes)
	areas = torchvision.ops.boxes.box_area(boxes)
	f = 1. / areas
	mat = inter * f.expand(inter.shape) - torch.eye(areas.shape[0])

	return torch.max(mat) > 0.4, areas
