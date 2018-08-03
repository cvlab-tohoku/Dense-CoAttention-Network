
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn

from torch.autograd import Variable


class LargeEmbedding(nn.Module):

	def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
				 norm_type=2, scale_grad_by_freq=False, sparse=False, devices=None):
		"""
		Store a large embedding matrix by spreading parts of it along GPU devies to save memory.
		Convert every index to a word vector.
		--------------------
		Arguments:
			num_embeddings (int): number of words in the embedding matrix.
			embedding_dim (int): dimension of word vectors.
			padding_idx (int): index of unk word.
			max_norm (int): norm of word vectors.
			norm_type (int): type of computed norm.
			scale_grad_by_freq (bool): If True, scale gradient of each word vectors by its frequency.
			sparse (bool): If True, store the embedding matrix in cpu.
			devices (list): the GPU devices that stores word vectors.
		"""
		super(LargeEmbedding, self).__init__()
		self.use_cuda = True if len(devices) > 0 else False
		self.devices = devices
		self.embedding_dim = embedding_dim
		self.num_embeddings = num_embeddings

		self.page_size = ((num_embeddings + len(devices) - 1) // len(devices)) if self.use_cuda else num_embeddings
		self.num_pages = len(devices) if self.use_cuda else 1

		self.embeddings = nn.ModuleList()
		for i in range(self.num_pages):
			self.embeddings.append(nn.Embedding(min(self.page_size, num_embeddings - i * self.page_size),
				embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse))

		if self.use_cuda:
			for i, embedding in enumerate(self.embeddings):
				embedding.cuda(self.devices[i])

	def load_pretrained_vectors(self, word_vectors, is_freeze=True):
		"""
		Loading the pretrained GloVe vectors to our embedding matrix.
		--------------------
		Arguments:
			word_vectors (string): path points to a ndarray file storing the GloVe vectors.
			is_freeze (bool): If True, all parameters are freezed during training.
		"""
		print("Loading pretrained word vectors from", word_vectors)
		pretrained = torch.load(word_vectors)["vectors"]

		for i, embedding in enumerate(self.embeddings):
			embedding.weight.data.copy_(pretrained[i*self.page_size: min((i+1)*self.page_size, self.num_embeddings)], 
				async=True)
			embedding.weight.requires_grad = False if is_freeze else True

	def forward(self, indices_):
		"""
		Arguments:
			indices_ (LongTensor: any shape): indices matrix where each index points to a word vector.
		Return:
			embedded (FloatTensor: any shape x embedding_dim): each index is replaced by a word vector.
		"""
		if self.use_cuda:
			indices = indices_.view(-1).cuda(self.devices[0])

			embedded = torch.FloatTensor(indices.size(0), self.embedding_dim).cuda(self.devices[0])
			idxs = torch.arange(0, indices.size(0)).long().cuda(self.devices[0])

			embedded = Variable(embedded)
			idxs = Variable(idxs)

			for i in range(self.num_pages):
				mask_i = torch.min(torch.ge(indices, i * self.page_size), torch.lt(indices, (i+1) * self.page_size))
				mask_idx = torch.masked_select(idxs, mask_i)
				if mask_idx.dim() == 0:
					continue

				indices_i = torch.index_select(indices, 0, mask_idx) - i * self.page_size
				indices_i = indices_i.cuda(self.devices[i])

				try:
					value_i = self.embeddings[i](indices_i).cuda(self.devices[0])
				except Exception:
					print("LargeEmbedding - %s, %s" % (indices_i, i * self.page_size))
					print("LargeEmbedding - %s" % self.devices[i])
					print("LargeEmbedding - %s" % self.embeddings[i])
					print("LargeEmbedding - %s" % indices_i.get_device()) if self.use_cuda else None
				embedded.index_copy_(0, mask_idx, value_i)

			embedded = embedded.view(indices_.size(0), indices_.size(1), self.embedding_dim)
		else:
			embedded = self.embeddings[0](indices_)

		return embedded