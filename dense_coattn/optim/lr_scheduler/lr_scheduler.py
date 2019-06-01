
from dense_coattn.optim import WrapOptimizer


class WrapLRScheduler(object):

	def __init__(self, args, optimizer):
		super(WrapLRScheduler, self).__init__()
		if not isinstance(optimizer, WrapOptimizer):
			raise ValueError("optimizer must be an instance of WrapOptimizer")
		self.args = args
		self.optimizer = optimizer
		self.best = None

	@staticmethod
	def add_args(parser):
		pass

	def state_dict(self):
		return {"best": self.best}

	def load_state_dict(self, state_dict):
		self.best = state_dict["best"]

	def step(self, epoch, val_acc=None):
		if val_acc is not None:
			if self.best is None:
				self.best = val_acc
			else:
				self.best = max(self.best, val_acc)

	def step_update(self, num_updates):
		return self.optimizer.get_lr()
