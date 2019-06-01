
import math

import torch


class WrapOptimizer(object):

	def __init__(self, args, params):
		super(WrapOptimizer, self).__init__()
		self.args = args
		params = list(params)
		if isinstance(params[0], dict):
			self.params = [p for p in group.values() for group in params]
		else:
			self.params = params

	@staticmethod
	def add_args(parser):
		pass

	@property
	def optimizer(self):
		if not hasattr(self, "_optimizer"):
			raise NotImplementedError
		if not isinstance(self._optimizer, torch.optim.Optimizer):
			raise ValueError("optimizer must be an instance of torch.optim.Optimizer")
		return self._optimizer

	@property
	def optimizer_config(self):
		raise NotImplementedError

	def get_stats(self):
		return self.optimizer.group_stats

	def get_lr(self):
		return self.optimizer.param_groups[0]["lr"]

	def set_lr(self, lr):
		for param_group in self.optimizer.param_groups:
			param_group["lr"] = lr

	def state_dict(self):
		return self.optimizer.state_dict()

	def load_state_dict(self, state_dict, optimizer_overrides=None):
		self.optimizer.load_state_dict(state_dict)

		if optimizer_overrides is not None and len(optimizer_overrides) > 0:
			for group in self.optimizer.param_groups:
				group.update(optimizer_overrides)

	def backward(self, loss):
		loss.backward()

	def multiply_grads(self, c):
		for p in self.params:
			if p.grad is not None:
				p.grad.data.mul_(c)

	def clip_grad_norm(self, max_norm):
		if max_norm > 0:
			return torch.nn.utils.clip_grad_norm_(self.params, max_norm)
		else:
			return math.sqrt(sum(p.grad.data.norm().item() ** 2 for p in self.params
							if p.grad is not None))

	def step(self, closure=None):
		self.optimizer.step(closure)

	def zero_grad(self):
		self.optimizer.zero_grad()
