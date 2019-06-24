
import torch


class Sampler(object):

	def __init__(self, data_source):
		pass

	def __iter__(self):
		raise NotImplementedError

	def __len__(self):
		raise NotImplementedError


class SequentialSampler(Sampler):

	def __init__(self, data_source):
		self.data_source = data_source

	def __iter__(self):
		return iter(range(len(self.data_source)))

	def __len__(self):
		return len(self.data_source)


class RandomSampler(Sampler):

	def __init__(self, data_source):
		self.data_source = data_source

	def __iter__(self):
		return iter(torch.randperm(len(self.data_source)).long())

	def __len__(self):
		return len(self.data_source)


class ComplementSampler(Sampler):

	def __init__(self, complement_idx):
		self.complement_idx = [idx for pair in complement_idx for idx in pair]

	def __iter__(self):
		return iter(self.complement_idx)

	def __len__(self):
		return len(self.complement_idx)


class SubsetRandomSampler(Sampler):

	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return (self.indices[i] for i in torch.randperm(len(self.indices)))

	def __len__(self):
		return len(self.indices)


class WeightedRandomSampler(Sampler):

	def __init__(self, weights, num_samples, replacement=True):
		self.weights = torch.DoubleTensor(weights)
		self.num_samples = num_samples
		self.replacement = replacement

	def __iter__(self):
		return iter(torch.multinomial(self.weights, self.num_samples, self.replacement))

	def __len__(self):
		return self.num_samples


class BatchSampler(object):

	def __init__(self, sampler, batch_size, drop_last):
		self.sampler = sampler
		self.batch_size = batch_size
		self.drop_last = drop_last

	def __iter__(self):
		batch = []
		for idx in self.sampler:
			batch.append(idx)
			if len(batch) == self.batch_size:
				yield batch
				batch = []

		if len(batch) > 0 and not self.drop_last:
			for _ in range(self.batch_size - len(batch)):
				batch.append(0)
			yield batch

	def __len__(self):
		if self.drop_last:
			return len(self.sampler) // self.batch_size
		else:
			return (len(self.sampler) + self.batch_size - 1) // self.batch_size
