
import torch
import torch.nn as nn
import torch.nn.functional as F

from .position_attn import PositionAttn
from .resnet import ResNet


class ImageExtractionLayer(nn.Module):

	def __init__(self, dim_vec, dim, num_attn, cnn_name="resnet152", is_freeze=True):
		super(ImageExtractionLayer, self).__init__()
		self.resnet = ResNet(cnn_name, is_freeze=is_freeze)
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
		self.extract4 = nn.Conv2d(2048, dim, kernel_size=1, stride=1)
		self.locate = PositionAttn(dim_vec, num_attn, 4)
		self.dim = dim
		
	def forward(self, img, ques_vec):
		batch = img.size(0)
		feat1, feat2, feat3, feat4 = self.resnet(img)
		feat1 = F.normalize(self.extract1(feat1), dim=1)
		feat2 = F.normalize(self.extract2(feat2), dim=1)
		feat3 = F.normalize(self.extract3(feat3), dim=1)
		feat4 = F.normalize(self.extract4(feat4), dim=1)

		feat = torch.stack([feat1, feat2, feat3, feat4], dim=1)
		feat = self.locate(ques_vec, feat).view(batch, self.dim, -1).transpose(1, 2).contiguous()
		
		return feat


class BottomUpExtract(nn.Module):

	def __init__(self, dim):
		super(BottomUpExtract, self).__init__()
		self.linear = nn.Linear(2048, dim)

	def forward(self, feat):
		feat = F.normalize(self.linear(feat), dim=2)

		return feat
