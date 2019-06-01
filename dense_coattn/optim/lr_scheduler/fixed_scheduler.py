
from .lr_scheduler import WrapLRScheduler


class FixedScheduler(WrapLRScheduler):

	def __init__(self, args, optimizer):
		super(FixedScheduler, self).__init__(args, optimizer)

		warmup_end_lr = args.lr[0]
		if args.warmup_init_lr < 0:
			args.warmup_init_lr = warmup_end_lr

		if args.warmup_updates > 0:
			self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_updates
		self.lr = args.warmup_init_lr
		self.optimizer.set_lr(self.lr)

	@staticmethod
	def add_args(parser):
		parser.add_argument("--force-anneal", "--fa", type=int,
							help="force annealing at specificed epoch")

	def get_next_lr(self, epoch):
		lrs = self.args.lr
		if self.args.force_anneal is None or epoch < self.args.force_anneal:
			next_lr = lrs[min(epoch, len(lrs) - 1)]
		else:
			next_lr = lrs[-1] * self.args.lr_shrink ** (epoch + 1 - self.args.force_anneal)
		
		return next_lr

	def step(self, epoch, val_acc=None):
		super().step(epoch, val_acc)
		self.lr = self.get_next_lr(epoch)
		self.optimizer.set_lr(self.lr)
		
		return self.optimizer.get_lr()

	def step_update(self, num_updates):
		if num_updates <= self.args.num_updates:
			self.lr = self.args.warmup_init_lr + num_updates * self.lr_step
			self.optimizer.set_lr(self.lr)

		return self.optimizer.get_lr()
