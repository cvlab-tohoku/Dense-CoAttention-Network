
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch.nn as nn
import torch.nn.functional as F


class PositionAttn(nn.Module):

	def __init__(self, dim, num_attn, num_loc):
		super(PositionAttn, self).__init__()

		self.linear = nn.Sequential(
				nn.Linear(dim, dim // 2),
				nn.ReLU(inplace=True),
				nn.Linear(dim // 2, num_attn * num_loc),
			)
		self.nlocs = num_loc
		self.h = num_attn
		self.attn = None

	def forward(self, query, value):
		batch, num_dim = query.size(0), value.dim()
		attn_shape = ([batch, self.nlocs] + [1] * (num_dim - 2))
		self.attn = F.softmax(self.linear(query).view(batch, self.h, self.nlocs), dim=2)
		w_attn = self.attn.mean(dim=1).view(*attn_shape)
		weighted = (value * w_attn).sum(dim=1)

		return weighted