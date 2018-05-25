
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn

from .self_attn import SummaryAttn


class PredictionLayer(nn.Module):

	def __init__(self, dim, num_attn, num_ans, predict_type, dropout, dropattn=0.1, is_cat=False):
		super(PredictionLayer, self).__init__()
		self.predict_type = predict_type
		self.summaries = nn.ModuleList([SummaryAttn(dim, num_attn, dropattn, is_cat=is_cat) for _ in range(2)])

		hidden_size = dim * num_attn if is_cat else dim
		if predict_type == "cat_attn":
			self.predict = nn.Sequential(
					nn.Linear(2 * hidden_size, hidden_size),
					nn.ReLU(inplace=True),
					nn.Dropout(p=dropout),
					nn.Linear(hidden_size, num_ans),
				)
		elif predict_type in ["sum_attn", "prod_attn"]:
			self.predict = nn.Sequential(
					nn.Linear(hidden_size, hidden_size),
					nn.ReLU(inplace=True),
					nn.Dropout(p=dropout),
					nn.Linear(hidden_size, num_ans),
				)
		else:
			raise TypeError("Invalid prediction type of output layer! \
							 Given {}, expected 'cat_attn', 'sum_attn', 'prod_attn'".format(predict_type))

	def forward(self, data1, data2, mask1, mask2):
		weighted1 = self.summaries[0](data1, data1, mask1)
		weighted2 = self.summaries[1](data2, data2, mask2)

		if self.predict_type == "cat_attn":
			weighted = torch.cat([weighted1, weighted2], dim=1)
		elif self.predict_type == "sum_attn":
			weighted = (weighted1 + weighted2)
		elif self.predict_type == "prod_attn":
			weighted = (weighted1 * weighted2)

		return self.predict(weighted)


class InnerPredictionLayer(nn.Module):

	def __init__(self, dim_vec, dim, num_attn, predict_type, dropattn=0.1, bias=False, is_cat=False):
		super(InnerPredictionLayer, self).__init__()
		self.predict_type = predict_type
		self.summaries = nn.ModuleList([SummaryAttn(dim, num_attn, dropattn, is_cat=is_cat) for _ in range(2)])

		hidden_size = dim * num_attn if is_cat else dim
		if predict_type in ["ans_prod_attn", "ans_sum_attn"]:
			self.linear = nn.Linear(dim_vec, hidden_size, bias=bias)
		elif predict_type == "ans_cat_attn":
			self.linear = nn.Linear(dim_vec, hidden_size*2, bias=bias)
		else:
			raise TypeError("Invalid prediction type of output layer! \
							 Given {}, expected 'ans_prod_attn', 'ans_sum_attn', 'ans_cat_attn'".format(predict_type))

	def forward(self, data1, data2, vec, mask1, mask2):
		weighted1 = self.summaries[0](data1, data1, mask1)
		weighted2 = self.summaries[1](data2, data2, mask2)

		if self.predict_type == "ans_prod_attn":
			weighted = weighted1 * weighted2
		elif self.predict_type == "ans_sum_attn":
			weighted = weighted1 + weighted2
		else:
			weighted = torch.cat([weighted1, weighted2], dim=1)

		return torch.mm(weighted, self.linear(vec).transpose(0, 1))