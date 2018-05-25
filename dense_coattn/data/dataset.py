
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import h5py
import random
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import PIL.Image as Image


type_map = {
	"train": "",
	"val": "",
	"test": "",
	"testdev": "",
}


class Dataset(object):

	def __init__(self, data_path, data_name, data_type, seq_per_img, img_name, size_scale=None, use_h5py=True):
		"""
		Load all of dataset information when initialize the Dataset object. 
		--------------------
		Parameters:
			data_path (str): path points to the folder containing data file.
			data_name (str): name of the dataset.
			data_type (str): "train", "val", "trainval", "test", or "testdev".
			seq_per_img (int): number of questions loaded with the same image.
			img_name (str): name of data file storing the image data.
			size_scale (int or tuple): If use_h5py is False, this is used to scale images.
			use_h5py (bool): If True, load image data from h5py instead of disk.
		"""
		img_type = data_type if data_type in ["train", "val", "trainval"] else "test"
		data_info = os.path.join(data_path, "%s_info.pt" % (data_name))
		data_file = os.path.join(data_path, "%s_%s.h5" % (data_name, data_type))
		img_file = os.path.join(data_path, "%s_%s.h5" % (img_name, img_type)) if use_h5py else None
		img_info = os.path.join(data_path, "%s_%s.pt" % (img_name, img_type)) if use_h5py else None

		self.data_type = data_type
		self.seq_per_img = seq_per_img
		self.data_info = torch.load(data_info)

		self.img2idx = torch.load(img_info) if use_h5py else None
		self.idx2word = self.data_info["idx2word"]
		self.idx2ans = self.data_info["idx2ans"]
		self.word2idx = self.data_info["word2idx"]
		self.ans2idx = self.data_info["ans2idx"]

		self.dataset = h5py.File(data_file, "r", driver="core")
		self.imageset = h5py.File(img_file, "r") if use_h5py else None

		self.images = self.imageset["images"][:] if use_h5py else None
		self.img_idx = self.dataset["img_idx"][:]
		self.txt_start_idx = self.dataset["txt_start_idx"][:]
		self.txt_end_idx = self.dataset["txt_end_idx"][:]
		self.questions = self.dataset["questions"][:]
		self.ques_idx = self.dataset["ques_idx"][:]
		self.ans_pool = self.dataset["ans_pool"][:]
		self.ans_idx = self.dataset["ans_idx"][:] if not data_type in ["test", "testdev"] else None

		if use_h5py:
			self.transform = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
				])
		else:
			self.transform = transforms.Compose([
				transforms.Resize(size_scale),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])

		print("Vocabulary size:", len(self.idx2word))
		print("Number of question:", self.img_idx.shape[0])
		print("Dataset %s_%s loaded" % (data_name, data_type))

	def __getitem__(self, idx):
		"""
		Arguments:
			idx (int): index of tuple (img, ques, ans) needed to extract from the dataset.
		Return:
			samples (tuple): contains image, question, question mask, index and answer score data.
		"""
		idx_start = self.txt_start_idx[idx]
		idx_end = self.txt_end_idx[idx]
		ntxt = idx_end - idx_start + 1
		assert ntxt > 0, "An image doesn't have any questions & answers!"

		img_idx = self.img_idx[idx]

		if self.img2idx is not None:
			img = self.transform(Image.fromarray(self.images[self.img2idx[img_idx]])).unsqueeze(0)
		else:
			if self.data_type != "trainval":
				img = self.transform(Image.open("%s_%.12i.jpg" % (type_map[self.data_type], img_idx))\
										.convert("RGB")).unsqueeze(0)
			else:
				if os.path.isfile("%s_%.12i.jpg" % (type_map["train"], img_idx)):
					img = self.transform(Image.open("%s_%.12i.jpg" % (type_map["train"], img_idx))\
											.convert("RGB")).unsqueeze(0)
				else:
					img = self.transform(Image.open("%s_%.12i.jpg" % (type_map["val"], img_idx))\
											.convert("RGB")).unsqueeze(0)

		if ntxt < self.seq_per_img:
			indices = [random.randint(idx_start, idx_end) for _ in range(ntxt)]
			ques = self.questions[indices]
			ques_idx = self.ques_idx[indices]
			ans_idx = self.ans_idx[indices] if self.ans_idx is not None else None
		else:
			index = random.randint(idx_start, idx_end - self.seq_per_img + 1)
			ques = self.questions[index:index+self.seq_per_img]
			ques_idx = self.ques_idx[index:index+self.seq_per_img]
			ans_idx = self.ans_idx[index:index+self.seq_per_img] if self.ans_idx is not None else None
		ques_mask = np.not_equal(ques, 0).astype(np.float32)

		if self.ans_idx is None:
			sample = (img, ques, ques_mask, ques_idx)
		else:
			sample = (img, ques, ques_mask, ques_idx, ans_idx)

		return sample

	def __len__(self):
		return self.questions.shape[0]
