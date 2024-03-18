import os
import argparse
import colorsys
import time
import json
import torch
import tqdm
import cv2
import numpy as np
import matplotlib
import matplotlib.image

from dataset_settings import EMNISTDetectionSettings
from box_functions import boxes_too_close, gen_boxes_from_segments
from segment_functions import extract_segments

parser = argparse.ArgumentParser()
parser.add_argument("--split", dest="split", default="train")
parser.add_argument("--ds-size", dest="dataset_size", default=1000, type=int)
parser.add_argument("--img-size", dest="img_size", default=320, type=int)
parser.add_argument("--target-dir", dest="target_dir", default="/home/USR/OD/emnist-detection-dataset-lfs")

torch.random.manual_seed(0)

class GenerateEMNISTDetection(object):
	def __init__(self,
	             image_size,
	             dataset_size,
				 split="train"
	             ):
		assert split in ["train", "val", "test"]
		self.split = split
		gen_from_train_set = (split in ["train", "val"])
		self.settings = EMNISTDetectionSettings(image_size=image_size, train=gen_from_train_set)
		self.dataset_size = dataset_size
		self.image_size = image_size
		empty_img_tensor = torch.ones(1, image_size, image_size)
		self.hsv_sampler = torch.distributions.uniform.Uniform(torch.tensor([0.0, 0.05, 0.1]), torch.tensor([1.0, 1.0, 1.0]))
		self.alpha_sampler = torch.distributions.uniform.Uniform(torch.tensor([0.5]), torch.tensor([0.9]))
		self.colorations = [
			lambda im: im.expand(3, image_size, image_size),
			lambda im: torch.cat((im.expand(2, image_size, image_size), empty_img_tensor), dim=0),
			lambda im: torch.cat((empty_img_tensor, im.expand(2, image_size, image_size)), dim=0),
			lambda im: torch.cat((im, empty_img_tensor, im), dim=0),
			lambda im: torch.cat((empty_img_tensor.expand(2, image_size, image_size), im), dim=0),
			lambda im: torch.cat((im, empty_img_tensor.expand(2, image_size, image_size)), dim=0),
			lambda im: torch.cat((empty_img_tensor, im, empty_img_tensor), dim=0),
		]

	def random_coloration(self, im):
		rnd_hsv = self.hsv_sampler.sample()
		rnd_rgb = colorsys.hsv_to_rgb(*rnd_hsv.tolist())
		im = 1 - im

		return torch.cat(((1 - (1-rnd_rgb[0])*im), (1 - (1-rnd_rgb[1])*im), (1 - (1-rnd_rgb[2])*im)), dim=0)

	def generate_set(self,
	                 target_directory,
	                 ):
  
		os.makedirs(os.path.join(target_directory, self.split, "img"), exist_ok=True)
		os.makedirs(os.path.join(target_directory, self.split, "annotations"), exist_ok=True)
		img_id = 0
		box_id = 0

		valid_data = []
		print("Gathering capital letters...")
		for data in tqdm.tqdm(self.settings.dataset):
			if data[1] in range(10, 36):
				valid_data.append(data)
		print(f"Found {len(valid_data)} capital letter data points.")

		images = []
		annotations = []
		tic = time.time()
		while img_id < self.dataset_size:
			img_tensors = []
			categories = []
			img_annotations = []
			num_instances = self.settings.poisson.sample()
			if num_instances == 0:
				bgd_dataset = self.settings.backgrounds
				if self.split == "val":
					bgd = bgd_dataset[len(bgd_dataset)-1-img_id][0]
				else:
					bgd = bgd_dataset[img_id][0]
				img_info, id_string = self.generate_image_info(img_id)
				images.append(img_info)
				img = bgd

			else:
				emnist_indices = torch.multinomial(torch.ones(len(valid_data)),
												num_samples=int(num_instances),
												replacement=True
												)
				for idx in emnist_indices:
					data = valid_data[idx]
					img_tensors.append(data[0].transpose(1, 2))
					categories.append(data[1] - 10)

				segs = extract_segments(img_tensors)
				boxes = gen_boxes_from_segments(segs)

				assert len(boxes) == len(segs)

				too_close_flag, areas = boxes_too_close(boxes)
				if too_close_flag:
					continue

				img = self.generate_image_combination(img_tensors)

				bgd_dataset = self.settings.backgrounds
				if self.split == "val":
					bgd = bgd_dataset[len(bgd_dataset)-1-img_id][0]
				else:
					bgd = bgd_dataset[img_id][0]

				if len(boxes):
					fg_bg_contrast_degenerate = False
					for b_count, b in enumerate(boxes):
						b = b.tolist()
						box = [b[0], b[1], b[2]-b[0], b[3]-b[1]]
						number_mask = img_tensors[b_count].long().bool()
						s = self.image_size
						r = torch.arange(s).expand(s, s)
						x = torch.logical_and(b[0] <= r, r <= b[2])
						y = torch.logical_and(b[1] <= r.transpose(0, 1), r.transpose(0, 1) <= b[3])
						box_mask = torch.logical_and(x, y)

						number_histos = []
						bgd_histos = []
						for i in range(3):
							number_hist = torch.histc(img[i][~number_mask[0]],
												min=0.0,
												max=1.0)
							bgd_hist = torch.histc(bgd[i][number_mask[0] & box_mask],
												min=0.0,
												max=1.0)
							number_histos.append(torch.argmax(number_hist))
							bgd_histos.append(torch.argmax(bgd_hist))

						if torch.norm(torch.stack(number_histos).float() - torch.stack(bgd_histos).float()) < 30.0/255:
							fg_bg_contrast_degenerate = True

						img_annotations.append(dict(segmentation=segs[b_count],
												area=areas[b_count].item(),
												iscrowd=0,
												image_id=img_id,
												bbox=box,
												category_id=categories[b_count]+1,
												id=box_id
												))
						box_id += 1

					if fg_bg_contrast_degenerate:
						box_id -= len(img_annotations)
						img_annotations = []
						continue

					bw_mask = torch.ones_like(img_tensors[0])
     
					for t in img_tensors:
						bw_mask = torch.minimum(bw_mask, t.round())
      
					bw_mask = bw_mask.expand(3, self.settings.image_size, self.settings.image_size)
     
					alpha = self.alpha_sampler.sample()
				img = torch.where(bw_mask.bool(), bw_mask*bgd, alpha*(1 - bw_mask)*img + (1 - alpha)*(1 - bw_mask)*bgd)
				img_info, id_string = self.generate_image_info(img_id)
				images.append(img_info)
				annotations.extend(img_annotations)
			if isinstance(img, torch.Tensor):
				img = np.array(img).transpose(1, 2, 0)
			k = np.ones((2, 2), np.float32)/4
			img = cv2.filter2D(img, -1, k)
			matplotlib.image.imsave(os.path.join(target_directory, self.split, "img", f"{id_string}.png"), img)

			img_id += 1

			if img_id % 1000 == 0:
				tac = time.time()
				print(f"prepared {img_id} images ({tac-tic} s).")
				tic = tac

		annotation_dict = {
			"info": self.settings.generate_info(),
			"licenses": self.settings.generate_licenses(),
			"images": images,
			"annotations": annotations,
			"categories": self.settings.generate_categories()
		}
		with open(os.path.join(target_directory,
		                       self.split,
		                       "annotations",
		                       f"emnistdetection_{self.split}_instances.json"), "w") as f:
			json.dump(annotation_dict, f)

		print("Dataset preparation finished.")

	def generate_image_info(self,
	                        img_id
	                        ):
		id_string = str(img_id).zfill(9)
		d = {
			"license": 0,
			"file_name": f"{id_string}.png",
			"coco_url": "",
			"height": self.settings.image_size,
			"width": self.settings.image_size,
			"date_captured": time.strftime("%Y-%m-%d %H:%M:%S"),
			"flickr_url": "",
			"id": img_id
		}

		return d, id_string

	def generate_image_combination(self,
	                               img_tensors):
		t = torch.ones_like(img_tensors[0])
		for img in img_tensors:
			t = torch.minimum(t, self.random_coloration(img))

		return t


if __name__ == "__main__":
	args = parser.parse_args()
	GenerateEMNISTDetection(image_size=args.img_size, dataset_size=args.dataset_size, split=args.split).generate_set(target_directory=args.target_dir)
