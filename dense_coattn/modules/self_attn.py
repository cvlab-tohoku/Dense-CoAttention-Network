
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn

from .attention_fn import sum_attention


class SummaryAttn(nn.Module):

	def __init__(self, dim, num_attn, dropout, is_cat=False):
		super(SummaryAttn, self).__init__()
		self.linear = nn.Sequential(
				nn.Linear(dim, dim),
				nn.ReLU(inplace=True),
				nn.Linear(dim, num_attn),
			)
		self.h = num_attn
		self.is_cat = is_cat
		self.attn = None
		self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

	def forward(self, query, value, mask=None):
		if mask is not None:
			mask = mask.unsqueeze(1)
		batch = query.size(0)

		weighted, self.attn = sum_attention(self.linear, query, value, mask=mask, dropout=self.dropout)
		weighted = weighted.view(batch, -1) if self.is_cat else weighted.mean(dim=1)

		return weighted