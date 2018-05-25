
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


def sum_attention(nnet, query, value, mask=None, dropout=None):
	scores = nnet(query).transpose(-2, -1)
	if mask is not None:
		scores.data.masked_fill_(mask.data.eq(0), -1e9)

	p_attn = F.softmax(scores, dim=-1)
	if dropout is not None:
		p_attn = dropout(p_attn)

	return torch.matmul(p_attn, value), p_attn


def qkv_attention(query, key, value, mask=None, dropout=None):
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(d_k)
	if mask is not None:
		scores.data.masked_fill_(mask.data.eq(0), -1e9)

	p_attn = F.softmax(scores, dim=-1)
	if dropout is not None:
		p_attn = dropout(p_attn)

	return torch.matmul(p_attn, value), p_attn