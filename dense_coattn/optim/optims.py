
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


class OptimWrapper(object):

	def __init__(self, optimizer, scheduler):
		self.optimizer = optimizer
		self.scheduler = scheduler

	def zero_grad(self):
		self.optimizer.zero_grad()

	def step(self):
		return self.optimizer.step()

	def step_epoch(self):
		self.scheduler.step()