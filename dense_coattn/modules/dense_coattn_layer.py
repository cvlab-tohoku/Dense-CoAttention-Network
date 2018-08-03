
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn

from .dense_coattn import DenseCoAttn


class NormalDenseSublayer(nn.Module):

	def __init__(self, dim, num_attn, num_none, dropout, dropattn=0.1):
		super(NormalDenseSublayer, self).__init__()
		self.dense_coattn = DenseCoAttn(dim, num_attn, num_none, dropattn)
		self.linears = nn.ModuleList([nn.Sequential(
				nn.Linear(2 * dim, dim),
				nn.ReLU(inplace=True),
				nn.Dropout(p=dropout),
			) for _ in range(2)])

	def forward(self, data1, data2, mask1, mask2):
		weighted1, weighted2 = self.dense_coattn(data1, data2, mask1, mask2)
		data1 = data1 + self.linears[0](torch.cat([data1, weighted2], dim=2))
		data2 = data2 + self.linears[1](torch.cat([data2, weighted1], dim=2))
		
		return data1, data2


class SimpleDCNLayer(nn.Module):

	def __init__(self, dim, num_attn, num_none, num_seq, dropout, dropattn=0.1):
		super(SimpleDCNLayer, self).__init__()
		self.dense_coattn_layers = nn.ModuleList([NormalDenseSublayer(dim, num_attn, num_none, dropout, dropattn=dropattn)
			for _ in range(num_seq)])

	def forward(self, data1, data2, mask1, mask2):
		for dense_coattn in self.dense_coattn_layers:
			data1, data2 = dense_coattn(data1, data2, mask1, mask2)

		return data1, data2