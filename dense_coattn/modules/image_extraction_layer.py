
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .position_attn import PositionAttn


class ImageExtractionLayer(nn.Module):

	def __init__(self, dim_vec, dim, num_attn, seq_per_img=0):
		super(ImageExtractionLayer, self).__init__()

		self.extract1 = nn.Sequential(
				nn.MaxPool2d(kernel_size=8, stride=8),
				nn.Conv2d(256, dim, kernel_size=1, stride=1),
			)
		self.extract2 = nn.Sequential(
				nn.MaxPool2d(kernel_size=4, stride=4),
				nn.Conv2d(512, dim, kernel_size=1, stride=1),
			)
		self.extract3 = nn.Sequential(
				nn.MaxPool2d(kernel_size=2, stride=2),
				nn.Conv2d(1024, dim, kernel_size=1, stride=1),
			)
		self.extract4 = nn.Sequential(
				nn.Conv2d(2048, dim, kernel_size=1, stride=1),
			)
		self.locate = PositionAttn(dim_vec, num_attn, 4)
		self.seq_per_img = seq_per_img

	def forward(self, feat1, feat2, feat3, feat4, ques_vec):
		feat1 = F.normalize(self.extract1(feat1), dim=1)
		feat2 = F.normalize(self.extract2(feat2), dim=1)
		feat3 = F.normalize(self.extract3(feat3), dim=1)
		feat4 = F.normalize(self.extract4(feat4), dim=1)

		feat = torch.stack([feat1, feat2, feat3, feat4], dim=1)
		if self.seq_per_img > 1:
			raise NotImplementedError

		return self.locate(ques_vec, feat)