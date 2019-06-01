
import math

from .lr_scheduler import WrapLRScheduler


class CosineScheduler(WrapLRScheduler):

	def __init__(self, args, optimizer):
		super(CosineScheduler, self).__init__(args, optimizer)
		if len(args.lr) > 1:
			raise ValueError(
				"Cannot use a fixed learning rate scheduler with CosineScheduler."
				" Consider --lr-scheduler=FixedScheduler instead."
			)
		
		warmup_end_lr = args.max_lr
		if args.warmup_init_lr < 0:
			args.warmup_init_lr = args.lr[0]

		self.min_lr = args.lr[0]
		self.max_lr = args.max_lr

		assert self.max_lr > self.min_lr, "max_lr must be greater than lr"

		self.t_mult = args.t_mult
		self.period = args.lr_period_updates

		if args.warmup_updates > 0:
			self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_updates

		self.warmup_updates = args.warmup_updates
		self.lr_shrink = args.lr_shrink

		self.lr = args.warmup_init_lr
		self.optimizer.set_lr(self.lr)

	@staticmethod
	def add_args(parser):
		parser.add_argument("--max-lr", required=True, type=float)
		parser.add_argument("--t-mult", default=1, type=float)
		parser.add_argument("--lr-period-updates", default=5000, type=float)

	def step(self, epoch, val_acc=None):
		super().step(epoch, val_acc)

		return self.optimizer.get_lr()

	def step_update(self, num_updates):
		if num_updates <= self.args.warmup_updates:
			self.lr = self.args.warmup_init_lr + num_updates * self.lr_step
		else:
			curr_updates = num_updates - self.warmup_updates
			if self.t_mult != 1:
				i = math.floor(math.log(1 - 
						curr_updates / self.period * (1 - self.t_mult), self.t_mult))
				t_i = self.t_mult ** i * self.period
				t_curr = curr_updates - \
						(1 - self.t_mult ** i) / (1 - self.t_mult) * self.period
			else:
				i = math.floor(curr_updates / self.period)
				t_i = self.period
				t_curr = curr_updates - (self.period * i)

			lr_shrink = self.lr_shrink ** i
			min_lr = self.min_lr * lr_shrink
			max_lr = self.max_lr * lr_shrink

			self.lr = min_lr + 0.5 * (max_lr - min_lr) * \
							(1 + math.cos(math.pi * t_curr / t_i))
		self.optimizer.set_lr(self.lr)
		
		return self.lr
