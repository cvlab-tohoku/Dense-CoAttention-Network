
from math import sqrt

import torch
import torch.nn.functional as F


def sum_attention(nnet, query, value, mask=None, dropout=None, mode='1D'):
	if mode == '2D':
		batch, dim = query.size(0), query.size(1)
		query = query.permute(0, 2, 3, 1).view(batch, -1, dim)
		value = value.permute(0, 2, 3, 1).view(batch, -1, dim)
		mask = mask.view(batch, 1, -1)

	scores = nnet(query).transpose(-2, -1)
	if mask is not None:
		scores.data.masked_fill_(mask.eq(0), -65504.0)

	p_attn = F.softmax(scores, dim=-1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	weighted = torch.matmul(p_attn, value)

	return weighted, p_attn


def qkv_attention(query, key, value, mask=None, dropout=None):
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2,-1)) / sqrt(d_k)
	if mask is not None:
		scores.data.masked_fill_(mask.eq(0), -65504.0)
	
	p_attn = F.softmax(scores, dim=-1)
	if dropout is not None:
		p_attn = dropout(p_attn)

	return torch.matmul(p_attn, value), p_attn
