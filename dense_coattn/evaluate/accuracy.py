
import torch
import torch.nn as nn


def evaluate(score, ans_idx):
	batch = score.size(0)

	_, inds = torch.sort(score, dim=1, descending=True)
	accuracy = torch.gather(ans_idx, 1, inds)[:, 0]
	accuracy = torch.sum(accuracy) * 100. / batch

	return accuracy
