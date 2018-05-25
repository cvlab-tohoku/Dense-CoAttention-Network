
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import torch

from torch.optim import Optimizer


class Adam(Optimizer):

	def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, record_step=10):
		defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, record_step=record_step)
		super(Adam, self).__init__(params, defaults)

	def step(self, closure=None):
		loss = None
		sum_abs_updates = 0.
		sum_abs_params = 0.
		ratio = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			for p in group["params"]:
				if p.grad is None:
					continue
				grad  = p.grad.data
				if grad.is_sparse:
					raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

				state = self.state[p]

				# state initialization
				if len(state) == 0:
					state["step"] = 0
					# Exponential moving average of gradient values
					state["exp_avg"] = torch.zeros_like(p.data)
					# Exponential moving average of squared gradient values
					state["exp_avg_sq"] = torch.zeros_like(p.data)

				exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
				beta1, beta2 = group["betas"]

				state["step"] += 1

				if group["weight_decay"] != 0:
					grad = grad.add(group["weight_decay"], p.data)

				# Decay the first and second mement running average coefficient
				exp_avg.mul_(beta1).add_(1 - beta1, grad)
				exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

				denom = exp_avg_sq.sqrt().add_(group["eps"])

				bias_correction1 = 1 - beta1 ** state["step"]
				bias_correction2 = 1 - beta2 ** state["step"]
				step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

				if state["step"] % group["record_step"] == 0:
					updates = (-step_size) * (exp_avg / denom)
					sum_abs_updates += torch.sum(torch.abs(updates))
					sum_abs_params += torch.sum(torch.abs(p.data))
					ratio = sum_abs_updates / (sum_abs_params + 1e-9)

				p.data.addcdiv_(-step_size, exp_avg, denom)

		return loss, ratio, sum_abs_updates, sum_abs_params