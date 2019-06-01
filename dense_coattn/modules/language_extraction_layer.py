
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class LSTM(nn.Module):

	def __init__(self, embed_size, dim, num_layers, dropout, residual_embeddings=True):
		super(LSTM, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		# self.rnn_dim = ((dim - embed_size) // 2) if residual_embeddings else (dim // 2)
		self.rnn_dim = 900 // 2
		self.linear = nn.Linear(900 + embed_size, dim)
		self.rnn = nn.LSTM(embed_size, self.rnn_dim, num_layers=num_layers, dropout=dropout,
			bidirectional=True, batch_first=True)
		self.residual_embeddings = residual_embeddings
		self.init_hidden = nn.Parameter(nn.init.xavier_uniform_(torch.empty(2*2*num_layers, self.rnn_dim)))
		self.num_layers = num_layers

	def forward(self, inputs, mask):
		batch = inputs.size(0)
		lengths = torch.sum(mask.data, dim=1)
		lens, indices = torch.sort(lengths, 0, True)
		_, _indices = torch.sort(indices, 0)

		lens = lens.type(torch.long).tolist()
		h0 = self.init_hidden[:2*self.num_layers].unsqueeze(1).expand(2*self.num_layers, 
									batch, self.rnn_dim).contiguous()
		c0 = self.init_hidden[2*self.num_layers:].unsqueeze(1).expand(2*self.num_layers, 
									batch, self.rnn_dim).contiguous()
		outputs, hidden_t = self.rnn(pack(inputs[indices], lens, batch_first=True), (h0, c0))

		outputs = unpack(outputs, batch_first=True)[0][_indices]
		hidden_t = (hidden_t[0].transpose(0, 1).contiguous().view(inputs.size(0), -1))[_indices]
		mask = unpack(pack(mask[indices], lens, batch_first=True), batch_first=True)[0][_indices]
		inputs = unpack(pack(inputs[indices], lens, batch_first=True), batch_first=True)[0][_indices]
		
		if self.residual_embeddings:
			outputs = torch.cat([inputs, outputs], 2)
		outputs = self.linear(self.dropout(outputs))

		return outputs, hidden_t, mask


class GRU(nn.Module):

	def __init__(self, embed_size, dim, num_layers, dropout, residual_embeddings=True):
		super(GRU, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		self.rnn_dim = 900 // 2
		self.linear = nn.Linear(900 + embed_size, dim)
		self.rnn = nn.GRU(embed_size, self.rnn_dim, num_layers=num_layers, dropout=dropout,
			bidirectional=True, batch_first=True)
		self.residual_embeddings = residual_embeddings
		self.init_hidden = nn.Parameter(nn.init.xavier_uniform_(torch.empty(2*num_layers, self.rnn_dim)))
		self.num_layers = num_layers

	def forward(self, inputs, mask):
		batch = inputs.size(0)
		lengths = torch.sum(mask.data, dim=1)
		lens, indices = torch.sort(lengths, 0, True)
		_, _indices = torch.sort(indices, 0)

		lens = lens.type(torch.long).tolist()
		h0 = self.init_hidden[:2*self.num_layers].unsqueeze(1).expand(2*self.num_layers, 
									batch, self.rnn_dim).contiguous()
		outputs, hidden_t = self.rnn(pack(inputs[indices], lens, batch_first=True), h0)

		outputs = unpack(outputs, batch_first=True)[0][_indices]
		hidden_t = (hidden_t[0].transpose(0, 1).contiguous().view(inputs.size(0), -1))[_indices]
		mask = unpack(pack(mask[indices], lens, batch_first=True), batch_first=True)[0][_indices]
		inputs = unpack(pack(inputs[indices], lens, batch_first=True), batch_first=True)[0][_indices]
		
		if self.residual_embeddings:
			outputs = torch.cat([inputs, outputs], 2)
		outputs = self.linear(self.dropout(outputs))

		return outputs, hidden_t, mask
