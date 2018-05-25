
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import h5py
import random
import numpy as np
import os
import torch


class RCNN_Dataset(object):

	def __init__(self, data_path, data_name, data_type, seq_per_img):
		img_type = "trainval" if data_type in ["train", "val", "trainval"] else "test"
		data_info = os.path.join(data_path, "%s_info.pt" % (data_name))
		data_file = os.path.join(data_path, "%s_%s.h5" % (data_name, data_type))
		img_file = os.path.join(data_path, "%s_images.h5" % (img_type))
		img_info = os.path.join(data_path, "%s_images.pt" % (img_type))

		self.data_type = data_type
		self.seq_per_img = seq_per_img
		self.data_info = torch.load(data_info)

		self.img2idx = torch.load(img_info)
		self.idx2word = self.data_info["idx2word"]
		self.idx2ans = self.data_info["idx2ans"]
		self.word2idx = self.data_info["word2idx"]
		self.ans2idx = self.data_info["ans2idx"]

		self.dataset = h5py.File(data_file, "r", driver="core")
		self.imageset = h5py.File(img_file, "r")

		self.features = self.imageset["features"][:]
		self.img_start_idx = self.imageset["img_start_idx"][:]
		self.img_end_idx = self.imageset["img_end_idx"][:]

		self.img_idx = self.dataset["img_idx"][:]
		self.txt_start_idx = self.dataset["txt_start_idx"][:]
		self.txt_end_idx = self.dataset["txt_end_idx"][:]
		self.questions = self.dataset["questions"][:]
		self.ques_idx = self.dataset["ques_idx"][:]
		self.ans_pool = self.dataset["ans_pool"][:]
		self.ans_idx = self.dataset["ans_idx"][:] if not data_type in ["test", "testdev"] else None

		print("Vocabulary size:", len(self.idx2word))
		print("Number of question:", self.img_idx.shape[0])
		print("Dataset %s_%s loaded" % (data_name, data_type))

	def __getitem__(self, idx):
		txt_start_idx = self.txt_start_idx[idx]
		txt_end_idx = self.txt_end_idx[idx]
		ntxt = txt_end_idx - txt_start_idx + 1
		assert ntxt > 0, "An image doesn't have any questions & answers!"

		img_idx = self.img2idx[self.img_idx[idx]]
		img_start_idx = self.img_start_idx[img_idx]
		img_end_idx = self.img_end_idx[img_idx]
		nboxes = img_end_idx - img_start_idx + 1

		img = np.zeros((1, 100, 2048), dtype=np.float32)
		img[0, :nboxes, :] = self.features[img_start_idx:img_end_idx + 1, :]

		if ntxt < self.seq_per_img:
			indices = [random.randint(txt_start_idx, txt_end_idx) for _ in range(ntxt)]
			ques = self.questions[indices]
			ques_idx = self.ques_idx[indices]
			ans_idx = self.ans_idx[indices] if self.ans_idx is not None else None
		else:
			index = random.randint(txt_start_idx, txt_end_idx - self.seq_per_img + 1)
			ques = self.questions[index:index+self.seq_per_img]
			ques_idx = self.ques_idx[index:index+self.seq_per_img]
			ans_idx = self.ans_idx[index:index+self.seq_per_img] if self.ans_idx is not None else None
		ques_mask = np.not_equal(ques, 0).astype(np.float32)
		img_mask = np.zeros((1, 100), dtype=np.float32)
		img_mask[0, :nboxes] = 1

		if self.ans_idx is None:
			sample = (img, ques, img_mask, ques_mask, ques_idx)
		else:
			sample = (img, ques, img_mask, ques_mask, ques_idx, ans_idx)

		return sample

	def __len__(self):
		return self.questions.shape[0]