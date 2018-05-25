
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


mtlstm_path = "/cove.pth" # Path point to CoVe parameters


class MTLSTM(nn.Module):

	def __init__(self, dim, dropout, residual_embeddings=True, is_freeze=False):
		super(MTLSTM, self).__init__()
		self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
		self.rnn_dim = (dim - 300) if residual_embeddings else dim
		self.linear = nn.Linear(600, self.rnn_dim) if self.rnn_dim != 600 else None
		self.rnn = nn.LSTM(300, 300, num_layers=2, dropout=dropout, bidirectional=True, batch_first=True)
		self.rnn.load_state_dict(torch.load(mtlstm_path))
		self.residual_embeddings = residual_embeddings
		self.init_hidden = nn.Embedding(2*2*2, 300)
		if is_freeze:
			print("Freezing MTLSTM layer...")
			for param in self.rnn.parameters():
				param.requires_grad = False

	def forward(self, inputs, mask):
		batch = inputs.size(0)
		lengths = torch.sum(mask, dim=1)
		lens, indices = torch.sort(lengths, 0, True)
		_, _indices = torch.sort(indices, 0)
		
		lens = lens.data.long().tolist()
		self.rnn.flatten_parameters()
		h0 = self.init_hidden(Variable(torch.arange(0, 2*2)).type_as(inputs)\
					.unsqueeze(1).long()).expand(2*2, batch, 300).contiguous()
		c0 = self.init_hidden(Variable(torch.arange(2*2, 2*2*2)).type_as(inputs)\
					.unsqueeze(1).long()).expand(2*2, batch, 300).contiguous()
		
		outputs, hidden_t = self.rnn(pack(inputs[indices.data], lens, batch_first=True), (h0, c0))
		outputs = unpack(outputs, batch_first=True)[0][_indices.data]
		
		hidden_t = (hidden_t[0].transpose(0, 1).contiguous().view(inputs.size(0), -1))[_indices.data]
		mask = unpack(pack(mask[indices.data], lens, batch_first=True), batch_first=True)[0][_indices.data]
		inputs = unpack(pack(inputs[indices.data], lens, batch_first=True), batch_first=True)[0][_indices.data]
		outputs = outputs if self.linear is None else self.linear(outputs)

		if self.residual_embeddings:
			outputs = torch.cat([inputs, outputs], 2)

		if self.dropout is not None:
			outputs = self.dropout(outputs)	
	
		return outputs, hidden_t, mask


class LSTM(nn.Module):

	def __init__(self, dim, num_layers, dropout, residual_embeddings=True):
		super(LSTM, self).__init__()
		print("Initializing Dynamic LSTM...")
		self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
		self.rnn_dim = ((dim - 300) // 2) if residual_embeddings else (dim // 2)
		self.rnn = nn.LSTM(300, self.rnn_dim, num_layers=num_layers, dropout=dropout, 
			bidirectional=True, batch_first=True)
		self.residual_embeddings = residual_embeddings
		self.init_hidden = nn.Embedding(2*2*num_layers, self.rnn_dim)
		self.num_layers = num_layers

	def forward(self, inputs, mask):
		batch = inputs.size(0)
		lengths = torch.sum(mask, dim=1)
		lens, indices = torch.sort(lengths, 0, True)
		_, _indices = torch.sort(indices, 0)
		
		lens = lens.data.long().tolist()
		self.rnn.flatten_parameters()
		h0 = self.init_hidden(Variable(torch.arange(0, 2*self.num_layers)).type_as(inputs)\
					.unsqueeze(1).long()).expand(2*self.num_layers, batch, self.rnn_dim).contiguous()
		c0 = self.init_hidden(Variable(torch.arange(2*self.num_layers, 2*2*self.num_layers)).type_as(inputs)\
					.unsqueeze(1).long()).expand(2*self.num_layers, batch, self.rnn_dim).contiguous()
		
		outputs, hidden_t = self.rnn(pack(inputs[indices.data], lens, batch_first=True), (h0, c0))
		outputs = unpack(outputs, batch_first=True)[0][_indices.data]
		
		hidden_t = (hidden_t[0].transpose(0, 1).contiguous().view(inputs.size(0), -1))[_indices.data]
		mask = unpack(pack(mask[indices.data], lens, batch_first=True), batch_first=True)[0][_indices.data]
		inputs = unpack(pack(inputs[indices.data], lens, batch_first=True), batch_first=True)[0][_indices.data]
		
		if self.residual_embeddings:
			outputs = torch.cat([inputs, outputs], 2)

		if self.dropout is not None:
			outputs = self.dropout(outputs)	
		
		return outputs, hidden_t, mask