
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch.nn as nn

from torch.autograd import Variable


class BinaryLoss(nn.Module):

	def __init__(self):
		super(BinaryLoss, self).__init__()
		self.loss = nn.BCEWithLogitsLoss(size_average=False)

	def forward(self, score, ans_idx):
		return self.loss(score, ans_idx)


class LossCompute(object):

	def __init__(self, criterion, opt=None):
		self.criterion = criterion
		self.opt = opt

	def __call__(self, score, ans_idx):
		loss = self.criterion(score, ans_idx)
		loss.backward()
		if self.opt is not None:
			self.opt.step()
			self.opt.zero_grad()

		return loss.data[0]