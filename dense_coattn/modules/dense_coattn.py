
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_fn import qkv_attention


class DenseCoAttn(nn.Module):

	def __init__(self, dim1, dim2, num_attn, num_none, dropout, is_multi_head=False):
		super(DenseCoAttn, self).__init__()
		dim = min(dim1, dim2)
		self.linears = nn.ModuleList([nn.Linear(dim1, dim, bias=False),
									  nn.Linear(dim2, dim, bias=False)])
		self.nones = nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_none, dim1))),
									   nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_none, dim2)))])
		self.d_k = dim // num_attn
		self.h = num_attn
		self.num_none = num_none
		self.is_multi_head = is_multi_head
		self.attn = None
		self.dropouts = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(2)])

	def forward(self, value1, value2, mask1=None, mask2=None):
		batch = value1.size(0)
		dim1, dim2 = value1.size(-1), value2.size(-1)
		value1 = torch.cat([self.nones[0].unsqueeze(0).expand(batch, self.num_none, dim1), value1], dim=1)
		value2 = torch.cat([self.nones[1].unsqueeze(0).expand(batch, self.num_none, dim2), value2], dim=1)
		none_mask = value1.new_ones((batch, self.num_none))

		if mask1 is not None:
			mask1 = torch.cat([none_mask, mask1], dim=1)
			mask1 = mask1.unsqueeze(1).unsqueeze(2)
		if mask2 is not None:
			mask2 = torch.cat([none_mask, mask2], dim=1)
			mask2 = mask2.unsqueeze(1).unsqueeze(2)

		query1, query2 = [l(x).view(batch, -1, self.h, self.d_k).transpose(1, 2) 
			for l, x in zip(self.linears, (value1, value2))]

		if self.is_multi_head:
			weighted1, attn1 = qkv_attention(query2, query1, query1, mask=mask1, dropout=self.dropouts[0])
			weighted1 = weighted1.transpose(1, 2).contiguous()[:, self.num_none:, :]
			weighted2, attn2 = qkv_attention(query1, query2, query2, mask=mask2, dropout=self.dropouts[1])
			weighted2 = weighted2.transpose(1, 2).contiguous()[:, self.num_none:, :]
		else:
			weighted1, attn1 = qkv_attention(query2, query1, value1.unsqueeze(1), mask=mask1, 
				dropout=self.dropouts[0])
			weighted1 = weighted1.mean(dim=1)[:, self.num_none:, :]
			weighted2, attn2 = qkv_attention(query1, query2, value2.unsqueeze(1), mask=mask2, 
				dropout=self.dropouts[1])
			weighted2 = weighted2.mean(dim=1)[:, self.num_none:, :]
		self.attn = [attn1[:,:,self.num_none:,self.num_none:], attn2[:,:,self.num_none:,self.num_none:]]

		return weighted1, weighted2
