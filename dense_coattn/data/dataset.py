
import collections
import os
import random

import h5py
import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class VisionLanguageDataset(Dataset):

	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)
		data_path = kwargs["data_path"]
		data_name = kwargs["data_name"]
		data_type = kwargs["data_type"]
		data_info_path = os.path.join(data_path, "{}_info.pt".format(data_name))
		data_file_path = os.path.join(data_path, "{}_{}.h5".format(data_name, data_type))
		self.data_info = torch.load(data_info_path)
		self.dataset = h5py.File(data_file_path, "r")

		image_path = kwargs["image_path"]
		image_type = kwargs["image_type"]
		image_split = kwargs["image_split"]
		if image_type == "rcnn":
			self.imageset = RCNNDataset(image_path, image_split)
		elif image_type == "resnet":
			self.imageset = ResnetDataset(image_path, image_split)
		else:
			raise TypeError("image dataset type should be 'rcnn, resnet, or fused_resnet' "
							"detected {}".format(image_type))


class RCNNDataset(Dataset):

	def __init__(self, data_path, data_type):
		super(RCNNDataset, self).__init__()
		img_info_path = os.path.join(data_path, "rcnn_{}.pt".format(data_type))
		self.img2idx = torch.load(img_info_path)

		self.img_file_path = os.path.join(data_path, "rcnn_{}.h5".format(data_type))
		self.imageset = None
		self.features = None
		self.img_start_idx = None
		self.img_end_idx = None

	def _initialize(self):
		if self.imageset is None or self.features is None or self.img_start_idx is None or self.img_end_idx is None:
			self.imageset = h5py.File(self.img_file_path, "r")
			self.features = self.imageset["features"]
			self.img_start_idx = self.imageset["img_start_idx"]
			self.img_end_idx = self.imageset["img_end_idx"]

	def __getitem__(self, index):
		self._initialize()
		img_idx = self.img2idx[index]
		img_start_idx = self.img_start_idx[img_idx]
		img_end_idx = self.img_end_idx[img_idx]
		nboxes = img_end_idx - img_start_idx + 1

		img = torch.zeros(1, 100, 2048, dtype=torch.float32)
		# img[0, :nboxes, :].copy_(torch.from_numpy(self.features[img_start_idx:img_end_idx+1,:]),
		#   non_blocking=True)
		img[0, :nboxes, :] = torch.from_numpy(self.features[img_start_idx:img_end_idx+1,:])
		img_mask = torch.zeros(1, 100, dtype=torch.float32)
		img_mask[0, :nboxes] = 1

		return (img, img_mask)

	def __len__(self):
		return len(self.img2idx)


class ResnetDataset(Dataset):

	def __init__(self, data_path, data_type):
		super(ResnetDataset, self).__init__()
		img_info_path = os.path.join(data_path, "resnet_{}.pt".format(data_type))
		self.img2idx = torch.load(img_info_path)
		self.transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
			])

		self.img_file_path = os.path.join(data_path, "resnet_{}.h5".format(data_type))
		self.imageset = None
		self.images = None
	
	def _initialize(self):
		if self.imageset is None or self.images is None:
			self.imageset = h5py.File(self.img_file_path, "r")
			self.images = self.imageset["images"]

	def __getitem__(self, index):
		self._initialize()
		img_idx = self.img2idx[index]
		img = self.transform(Image.fromarray(self.images[img_idx])).unsqueeze(0)

		return img, None

	def __len__(self):
		return len(self.img2idx)


class VQADataset(VisionLanguageDataset):

	def __init__(self, data_path, data_name, data_type, image_path, image_type, image_split):
		super(VQADataset, self).__init__(data_path=data_path, data_name=data_name, 
			data_type=data_type, image_path=image_path, image_type=image_type, image_split=image_split)

		self.idx2word = self.data_info["idx2word"]
		self.idx2ans = self.data_info["idx2ans"]
		self.word2idx = self.data_info["word2idx"]
		self.and2idx = self.data_info["ans2idx"]

		self.img_idx = self.dataset["img_idx"][:]
		self.questions = self.dataset["questions"][:]
		self.ques_idx = self.dataset["ques_idx"][:]
		self.ans_pool = self.dataset["ans_pool"][:]
		
		if "test" in data_type:
			self.ans_idx = None
		else:
			self.ans_idx = self.dataset["ans_idx"][:]
		print(f"Initializing {data_type} VQA dataset: {self.questions.shape[0]} questions, {len(self.imageset)} images")

	def __getitem__(self, index):
		ques = torch.from_numpy(self.questions[[index]])
		ques_idx = torch.from_numpy(self.ques_idx[[index]])
		ans_idx = torch.from_numpy(self.ans_idx[[index]]) if self.ans_idx is not None else None
		ques_mask = ques.ne(0).float()
		img_info = self.imageset[self.img_idx[index]]

		return (img_info, ques, ques_mask, ans_idx, ques_idx)

	def __len__(self):
		return self.questions.shape[0]
