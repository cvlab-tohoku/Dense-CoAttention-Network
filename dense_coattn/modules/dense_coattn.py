
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn

from torch.autograd import Variable
from .attention_fn import qkv_attention


class DenseCoAttn(nn.Module):

	def __init__(self, dim, num_attn, num_none, dropout, is_cat=False):
		super(DenseCoAttn, self).__init__()
		self.linears = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(2)])
		self.nones = nn.ModuleList([nn.Embedding(num_none, dim) for _ in range(2)])
		self.d_k = dim // num_attn
		self.h = num_attn
		self.num_none = num_none
		self.is_cat = is_cat
		self.attn = None
		self.dropouts = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(2)]) if dropout > 0 else [None, None]

	def forward(self, value1, value2, mask1=None, mask2=None):
		batch = value1.size(0)
		none_mask = None
		none = Variable(torch.arange(self.num_none).unsqueeze(0)).type_as(value1).long().expand(batch, self.num_none) \
			if self.num_none > 0 else None
		if none is not None:
			value1 = torch.cat([self.nones[0](none), value1], dim=1)
			value2 = torch.cat([self.nones[1](none), value2], dim=1)
			none_mask = Variable(torch.ones(batch, self.num_none)).type_as(value1)

		if mask1 is not None:
			mask1 = mask1 if none_mask is None else torch.cat([none_mask, mask1], dim=1)
			mask1 = mask1.unsqueeze(1).unsqueeze(2)
		if mask2 is not None:
			mask2 = mask2 if none_mask is None else torch.cat([none_mask, mask2], dim=1)
			mask2 = mask2.unsqueeze(1).unsqueeze(2)

		query1, query2 = \
			[l(x).view(batch, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (value1, value2))]

		if self.is_cat:
			weighted1, attn1 = qkv_attention(query2, query1, query1, mask=mask1, dropout=self.dropouts[0])
			weighted1 = weighted1.transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_k)[:, self.num_none:, :]
			weighted2, attn2 = qkv_attention(query1, query2, query2, mask=mask2, dropout=self.dropouts[1])
			weighted2 = weighted2.transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_k)[:, self.num_none:, :]
		else:
			weighted1, attn1 = qkv_attention(query2, query1, value1.unsqueeze(1), mask=mask1, dropout=self.dropouts[0])
			weighted1 = weighted1.mean(dim=1)[:, self.num_none:, :]
			weighted2, attn2 = qkv_attention(query1, query2, value2.unsqueeze(1), mask=mask2, dropout=self.dropouts[1])
			weighted2 = weighted2.mean(dim=1)[:, self.num_none:, :]
		self.attn = [attn1[:,:,self.num_none:,self.num_none:], attn2[:,:,self.num_none:,self.num_none:]]

		return weighted1, weighted2