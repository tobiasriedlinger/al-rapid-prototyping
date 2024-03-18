import time
import string
import torch
import torchvision
import torchvision.transforms as transforms

class EMNISTDetectionSettings(object):

	def __init__(self, image_size=320, train=True):
		self.date_created = time.strftime(("%Y/%m/%d"))
		self.image_size = image_size
		self.transform = transforms.Compose([
			transforms.Resize(image_size),
			transforms.RandomAffine(5.0, translate=(0.4, 0.4), scale=(0.1, 0.35), shear=3.0),
			transforms.RandomInvert(1.0),
			transforms.ToTensor()
		])
		self.dataset = torchvision.datasets.EMNIST(
			root="/home/USR/active_learning_od/nutshell_emnist/EMNIST_base",
			split="byclass",
			train=train,
			download=True,
			transform=self.transform
			)

		background_transform = transforms.Compose([
			transforms.Resize(image_size),
			transforms.CenterCrop(image_size),
			transforms.ToTensor()
		])
		bgd_id = "train" if train else "val"
		self.backgrounds = torchvision.datasets.CocoDetection(
			root=f"/home/USR/datasets/COCO/2017/{bgd_id}2017",
			annFile=f"/home/datasets/USR/COCO/2017/annotations/instances_{bgd_id}2017.json",
			transform=background_transform,
		)

		self.poisson_average = 4
		self.poisson = torch.distributions.poisson.Poisson(rate=self.poisson_average)

		self.multinom_weights = torch.ones(len(self.dataset))

	def generate_info(self):
		description = "EMNISTDetection Dataset"
		url = ""
		version = "1.0"
		year = "2022"
		contributor = ""

		return {"description": description,
		        "url": url,
		        "version": version,
		        "year": year,
		        "contributor": contributor,
		        "date_created": self.date_created
		        }

	def generate_licenses(self):
		return [
			{
				"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
				"id": 1,
				"name": "Attribution-NonCommercial-ShareAlike License"
			}]

	def generate_categories(self):
		letters = list(string.ascii_letters.split("z")[-1])
		return [{"supercategory": "emnist_letter",
		         "id": i+1,
		         "name": letters[i]} for i in range(len(letters))
		        ]
