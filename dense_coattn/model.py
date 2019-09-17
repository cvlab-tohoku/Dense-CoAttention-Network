
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .modules import *
from .utils import Initializer


class DCN(nn.Module):

	def __init__(self, opt, num_ans):
		super(DCN, self).__init__()
		self.lang_extract = LSTM(300, opt.ques_size, opt.num_layers, opt.droprnn, residual_embeddings=True)
		
		rnn_dim = (opt.ques_size - 300)
		self.img_extract = ImageExtractionLayer(opt.num_layers*rnn_dim, opt.img_size, 
			opt.num_img_attn, cnn_name=opt.cnn_name)
		self.dense_coattn = DCNLayer(opt.img_size, opt.ques_size, opt.num_dense_attn, opt.num_none, 
			opt.num_seq, opt.dropout)
		self.predict = PredictLayer(opt.img_size, opt.ques_size, opt.num_predict_attn, num_ans, opt.dropout)
		self.apply(Initializer.xavier_normal)

	def forward(self, img, ques, img_mask, ques_mask):
		ques, ques_vec, ques_mask = self.lang_extract(ques, ques_mask)
		img = self.img_extract(img, ques_vec)

		img, ques = self.dense_coattn(img, ques, img_mask, ques_mask)
		score = self.predict(img, ques, img_mask, ques_mask)

		return score


class DCNWithRCNN(nn.Module):

	def __init__(self, opt, num_ans):
		super(DCNWithRCNN, self).__init__()
		print("Initializing DCNWithRCNN...")
		self.lang_extract = LSTM(300, opt.ques_size, opt.num_layers, opt.droprnn, residual_embeddings=True)
		self.img_extract = BottomUpExtract(opt.img_size)
		self.dense_coattn = DCNLayer(opt.img_size, opt.ques_size, opt.num_dense_attn, opt.num_none, 
			opt.num_seq, opt.dropout)
		self.predict = PredictLayer(opt.img_size, opt.ques_size, opt.num_predict_attn, num_ans, opt.dropout)
		self.apply(Initializer.xavier_normal)

	def forward(self, img, ques, img_mask, ques_mask):
		ques, _, ques_mask = self.lang_extract(ques, ques_mask)
		img = self.img_extract(img)

		img, ques = self.dense_coattn(img, ques, img_mask, ques_mask)
		score = self.predict(img, ques, img_mask, ques_mask)

		return score
