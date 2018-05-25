
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch

from torch.optim import Optimizer


class SGD(Optimizer):

	def __init__(self, params, lr=0.1, momentum=0, dampening=0, weight_decay=0, nesterov=False, record_step=10):
		defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov,
			record_step=record_step)
		if nesterov and (momentum <= 0 or dampening != 0):
			raise ValueError("Nesterov momentum reuquires a momentum and zero dampening")
		super(SGD, self).__init__(params, defaults)

	def __setstate__(self, state):
		super(SGD, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault("nesterov", False)

	def step(self, closure=None):
		loss = None
		sum_abs_update = 0.
		sum_abs_params = 0.
		ratio = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			weight_decay = group["weight_decay"]
			momentum = group["momentum"]
			dampening = group["dampening"]
			nesterov = group["nesterov"]

			for p in group["params"]:
				if p.grad is None:
					continue
				d_p = p.grad.data
				state = self.state[p]
				if len(state) == 0:
					state["step"] = 0

				state["step"] += 1

				if weight_decay != 0:
					d_p.add_(weight_decay, p.data)
				if momentum != 0:
					param_state = self.state[p]
					if "momentum_buffer" not in param_state:
						buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
						buf.mul_(momentum).add_(d_p)
					else:
						buf = param_state["momentum_buffer"]
						buf.mul_(momentum).add_(1 - dampening, d_p)
					if nesterov:
						d_p = d_p.add(momentum, buf)
					else:
						d_p = buf

				if state["step"] % group["record_step"] == 0:
					updates = (-group["lr"]) * d_p
					sum_abs_update += torch.sum(torch.abs(updates))
					sum_abs_params += torch.sum(torch.abs(p.data))
					ratio = sum_abs_update / (sum_abs_params + 1e-9)

				p.data.add_(-group["lr"], d_p)

		return loss, ratio, sum_abs_update, sum_abs_params