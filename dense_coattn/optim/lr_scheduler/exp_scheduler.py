
import torch.optim.lr_scheduler

from .lr_scheduler import WrapLRScheduler


class ExponentialLR(WrapLRScheduler):

	def __init__(self, args, optimizer):
		super(ExponentialLR, self).__init__(args, optimizer)
		if len(args.lr) > 1:
			raise ValueError(
				"Cannot use a fixed learning rate scheduler with ExponentialLR."
				" Consider --lr-scheduler=FixedScheduler instead."
			)
		self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
			self.optimizer.optimizer, gamma=args.lr_shrink
		)
		warmup_end_lr = args.lr[0]
		if args.warmup_init_lr < 0:
			args.warmup_init_lr = warmup_end_lr

		if args.warmup_updates > 0:
			self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_updates

		self.lr = args.warmup_init_lr
		self.optimizer.set_lr(self.lr)

	def state_dict(self):
		return {
			"best": self.best,
			"last_epoch": self.lr_scheduler.last_epoch,
		}
	
	def load_state_dict(self, state_dict):
		self.best = state_dict["best"]
		self.lr_scheduler.last_epoch = state_dict["last_epoch"]
	
	def step(self, epoch, val_acc=None):
		super().step(epoch, val_acc)
		self.lr_scheduler.step(epoch)
		self.lr = self.optimizer.get_lr()

		return self.optimizer.get_lr()

	def step_update(self, num_updates):
		if num_updates <= self.args.warmup_updates:
			self.lr = self.args.warmup_init_lr + num_updates * self.lr_step
			self.optimizer.set_lr(self.lr)

		return self.optimizer.get_lr()
