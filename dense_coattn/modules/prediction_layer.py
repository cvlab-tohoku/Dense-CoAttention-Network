
import torch
import torch.nn as nn

from .self_attn import SummaryAttn


class PredictLayer(nn.Module):

	def __init__(self, dim1, dim2, num_attn, num_ans, dropout, dropattn=0):
		super(PredictLayer, self).__init__()
		self.summaries = nn.ModuleList([
			SummaryAttn(dim1, num_attn, dropattn, is_multi_head=False),
			SummaryAttn(dim2, num_attn, dropattn, is_multi_head=False),
		])

		self.predict = nn.Sequential(
			nn.Linear(dim1 + dim2, (dim1 + dim2) // 2),
			nn.ReLU(inplace=True),
			nn.Dropout(p=dropout),
			nn.Linear((dim1 + dim2) // 2, num_ans),
		)

	def forward(self, data1, data2, mask1, mask2):
		weighted1 = self.summaries[0](data1, data1, mask1)
		weighted2 = self.summaries[1](data2, data2, mask2)
		weighted = torch.cat([weighted1, weighted2], dim=1)

		return self.predict(weighted)
