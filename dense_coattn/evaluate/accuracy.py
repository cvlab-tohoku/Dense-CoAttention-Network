
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn

from torch.autograd import Variable


class Accuracy(nn.Module):

	def __init__(self):
		super(Accuracy, self).__init__()

	def forward(self, score, ans_idx):
		batch = score.size(0)

		_, inds = torch.sort(score, dim=1, descending=True)
		accuracy = torch.gather(ans_idx, 1, inds)[:, 0]
		accuracy = torch.sum(accuracy) * 100. / batch

		return accuracy