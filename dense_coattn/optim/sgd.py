
import torch
import torch.optim
from torch.optim import Optimizer

from .optimizer import WrapOptimizer


class SGD(WrapOptimizer):

	def __init__(self, args, params):
		super(SGD, self).__init__(args, params)
		self._optimizer = FixedSGD(params, **self.optimizer_config)

	@property
	def optimizer_config(self):
		return {
			"lr": self.args.lr[0],
			"momentum": self.args.momentum,
			"weight_decay": self.args.weight_decay,
		}


class FixedSGD(Optimizer):

	def __init__(self, params, lr=0.1, momentum=0, dampening=0, weight_decay=0, nesterov=False, record=True):
		defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov,
			record=record)
		if nesterov and (momentum <= 0 or dampening != 0):
			raise ValueError("Nesterov momentum reuquires a momentum and zero dampening")
		super(FixedSGD, self).__init__(params, defaults)
		self.group_stats = []

	def __setstate__(self, state):
		super(FixedSGD, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault("nesterov", False)

	def step(self, closure=None):
		loss = None
		self.group_stats = []
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			group_abs_updates = 0.
			group_abs_params = 0.
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

				if group["record"]:
					updates = (-group["lr"]) * d_p
					group_abs_updates += torch.sum(torch.abs(updates))
					group_abs_params += torch.sum(torch.abs(p.data))

				p.data.add_(-group["lr"], d_p)
			self.group_stats.append((group_abs_updates, group_abs_params))

		return loss
