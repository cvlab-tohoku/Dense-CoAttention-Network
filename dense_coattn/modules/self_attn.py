
import torch
import torch.nn as nn

from .attention_fn import sum_attention


class SummaryAttn(nn.Module):

	def __init__(self, dim, num_attn, dropout, is_multi_head=False, mode='1D'):
		super(SummaryAttn, self).__init__()
		self.linear = nn.Sequential(
			nn.Linear(dim, dim),
			nn.ReLU(inplace=True),
			nn.Linear(dim, num_attn),
		)
		self.h = num_attn
		self.is_multi_head = is_multi_head
		self.attn = None
		self.dropout = nn.Dropout(p=dropout) if dropout else None
		self.mode = mode

	def forward(self, query, value, mask=None):
		if mask is not None:
			mask = mask.unsqueeze(1)
		batch = query.size(0)

		weighted, self.attn = sum_attention(self.linear, query, value, mask=mask, dropout=self.dropout, mode=self.mode)
		weighted = weighted.view(batch, -1) if self.is_multi_head else weighted.mean(dim=-2)

		return weighted
