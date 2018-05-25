
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from .modules import *
from .util import Initializer


class DCN(nn.Module):

	def __init__(self, opt, num_ans):
		super(DCN, self).__init__()
		self.lang_extract = LSTM(opt.hidden_size, opt.num_layers, opt.droprnn, residual_embeddings=True)
		self.resnet = ResNet(opt.cnn_name, is_freeze=True)
		
		rnn_dim = (opt.hidden_size - 300)
		self.img_extract = ImageExtractionLayer(opt.num_layers*rnn_dim, opt.hidden_size, 
			opt.num_img_attn, opt.seq_per_img)
		self.dense_coattn = SimpleDCNLayer(opt.hidden_size, opt.num_dense_attn, opt.num_none, 
			opt.num_seq, opt.dropout, dropattn=opt.dropattn)
		self.predict = PredictionLayer(opt.hidden_size, opt.num_predict_attn, num_ans, opt.predict_type, 
			opt.dropout, dropattn=opt.dropattn, is_cat=False)

		self.hidden_size = opt.hidden_size
		self.apply(Initializer.xavier_normal)

	def forward(self, img, ques, img_mask, ques_mask, is_train=True):
		batch = ques.size(0)
		feat1, feat2, feat3, feat4 = self.resnet(img)
		if is_train:
			feat1 = Variable(feat1.data)
			feat2 = Variable(feat2.data)
			feat3 = Variable(feat3.data)
			feat4 = Variable(feat4.data)

		ques, ques_vec, ques_mask = self.lang_extract(ques, ques_mask)
		img = self.img_extract(feat1, feat2, feat3, feat4, ques_vec)
		img = img.view(batch, self.hidden_size, -1).transpose(1, 2).contiguous()

		img, ques = self.dense_coattn(img, ques, img_mask, ques_mask)
		score = self.predict(img, ques, img_mask, ques_mask)

		return score


class DCNWithAns(nn.Module):

	def __init__(self, opt, ans, ans_mask):
		super(DCNWithAns, self).__init__()
		self.lang_extract = LSTM(opt.hidden_size, opt.num_layers, opt.droprnn, residual_embeddings=True)
		self.resnet = ResNet(opt.cnn_name, is_freeze=True)
		
		rnn_dim = (opt.hidden_size - 300)
		self.img_extract = ImageExtractionLayer(opt.num_layers*rnn_dim, opt.hidden_size, 
			opt.num_img_attn, opt.seq_per_img)
		self.dense_coattn = SimpleDCNLayer(opt.hidden_size, opt.num_dense_attn, opt.num_none, 
			opt.num_seq, opt.dropout, dropattn=opt.dropattn)
		self.predict = InnerPredictionLayer(opt.num_layers*rnn_dim, opt.hidden_size, opt.num_predict_attn, 
			opt.predict_type, dropattn=opt.dropattn, bias=False, is_cat=False)

		self.ans = nn.Parameter(ans, requires_grad=False)
		self.ans_mask = nn.Parameter(ans_mask, requires_grad=False)
		self.hidden_size = opt.hidden_size
		self.apply(Initializer.xavier_normal)

	def forward(self, img, ques, img_mask, ques_mask, is_train=True):
		batch = ques.size(0)
		num_ans = self.ans.size(0)
		ans = Variable(self.ans.data)
		ans_mask = Variable(self.ans_mask.data)
		feat1, feat2, feat3, feat4 = self.resnet(img)
		if is_train:
			feat1 = Variable(feat1.data)
			feat2 = Variable(feat2.data)
			feat3 = Variable(feat3.data)
			feat4 = Variable(feat4.data)

		ques, ques_vec, ques_mask = self.lang_extract(ques, ques_mask)
		_, ans_vec, _ = self.lang_extract(ans, ans_mask)
		img = self.img_extract(feat1, feat2, feat3, feat4, ques_vec)
		img = img.view(batch, self.hidden_size, -1).transpose(1, 2).contiguous()

		img, ques = self.dense_coattn(img, ques, img_mask, ques_mask)
		score = self.predict(img, ques, ans_vec, img_mask, ques_mask)

		return score


class DCNWithRCNN(nn.Module):

	def __init__(self, opt, num_ans):
		super(DCNWithRCNN, self).__init__()
		self.lang_extract = LSTM(opt.hidden_size, opt.num_layers, opt.droprnn, residual_embeddings=True)
		self.img_extract = nn.Linear(2048, opt.hidden_size)

		self.dense_coattn = SimpleDCNLayer(opt.hidden_size, opt.num_dense_attn, opt.num_none, 
			opt.num_seq, opt.dropout, dropattn=opt.dropattn)
		self.predict = PredictionLayer(opt.hidden_size, opt.num_predict_attn, num_ans, opt.predict_type, 
			opt.dropout, dropattn=opt.dropattn, is_cat=False)

		self.hidden_size = opt.hidden_size
		self.apply(Initializer.xavier_normal)

	def forward(self, img, ques, img_mask, ques_mask):
		batch = ques.size(0)
		ques, _, ques_mask = self.lang_extract(ques, ques_mask)
		img = F.normalize(self.img_extract(img), dim=2)

		img, ques = self.dense_coattn(img, ques, img_mask, ques_mask)
		score = self.predict(img, ques, img_mask, ques_mask)

		return score


class DCNWithRCNNAns(nn.Module):

	def __init__(self, opt, ans, ans_mask):
		super(DCNWithRCNNAns, self).__init__()
		self.lang_extract = LSTM(opt.hidden_size, opt.num_layers, opt.droprnn, residual_embeddings=True)
		self.img_extract = nn.Linear(2048, opt.hidden_size)

		rnn_dim = (opt.hidden_size - 300)
		self.dense_coattn = SimpleDCNLayer(opt.hidden_size, opt.num_dense_attn, opt.num_none, 
			opt.num_seq, opt.dropout, dropattn=opt.dropattn)
		self.predict = InnerPredictionLayer(opt.num_layers*rnn_dim, opt.hidden_size, opt.num_predict_attn, 
			opt.predict_type, dropattn=opt.dropattn, bias=False, is_cat=False)

		self.ans = nn.Parameter(ans, requires_grad=False)
		self.ans_mask = nn.Parameter(ans_mask, requires_grad=False)
		self.hidden_size = opt.hidden_size
		self.apply(Initializer.xavier_normal)

	def forward(self, img, ques, img_mask, ques_mask):
		batch = ques.size(0)
		num_ans = self.ans.size(0)
		ans = Variable(self.ans.data)
		ans_mask = Variable(self.ans_mask.data)
		ques, _, ques_mask = self.lang_extract(ques, ques_mask)
		_, ans_vec, _ = self.lang_extract(ans, ans_mask)
		img = F.normalize(self.img_extract(img), dim=2)

		img, ques = self.dense_coattn(img, ques, img_mask, ques_mask)
		score = self.predict(img, ques, ans_vec, img_mask, ques_mask)

		return score