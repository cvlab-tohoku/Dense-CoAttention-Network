
from .lr_scheduler import WrapLRScheduler


class InverseSquareRootScheduler(WrapLRScheduler):

	def __init__(self, args, optimizer):
		super(InverseSquareRootScheduler, self).__init__(args, optimizer)
		if len(args.lr) > 1:
			raise ValueError(
				"Cannot use a fixed learning rate scheduler with InverseSquareRootScheduler."
				" Consider --lr-scheduler=FixedScheduler instead."
			)
		warmup_end_lr = args.lr[0]
		if args.warmup_init_lr < 0:
			args.warmup_init_lr = warmup_end_lr

		# linearly warmup for the first args.warmup_updates
		if args.warmup_updates > 0:
			self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_updates

		# then, decay prop. to the inverse square root of the update number
		self.decay_factor = warmup_end_lr * args.warmup_updates ** 0.5

		# initial learning rate
		self.lr = args.warmup_init_lr
		self.optimizer.set_lr(self.lr)

	def step(self, epoch, val_loss=None):
		super().step(epoch, val_loss)
		# don't change the learning rate at epoch boundaries
		return self.optimizer.get_lr()

	def step_update(self, num_updates):
		if num_updates < self.args.warmup_updates:
			self.lr = self.args.warmup_init_lr + num_updates * self.lr_step
		else:
			self.lr = self.decay_factor * num_updates ** (-0.5)
		self.optimizer.set_lr(self.lr)

		return self.lr
