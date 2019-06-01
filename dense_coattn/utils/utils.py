
import os
import random
import shutil
import time
from collections import Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.misc as misc
import skimage.transform as transform
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont


class Initializer(object):

	@staticmethod
	def manual_seed(seed):
		"""
		Set all of random seed to seed.
		--------------------
		Arguments:
			seed (int): seed number.
		"""
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)

	@staticmethod
	def xavier_normal(module, lstm_forget_bias_init=2):
		"""
		Xavier Gaussian initialization.
		"""
		lstm_forget_bias_init = float(lstm_forget_bias_init) / 2
		normal_classes = (nn.Conv2d, nn.Linear, nn.Embedding)
		recurrent_classes = (nn.RNN, nn.LSTM, nn.GRU)
		if any([isinstance(module, cl) for cl in normal_classes]):
			nn.init.xavier_normal_(module.weight.data) if module.weight.requires_grad else None
			try:
				module.bias.data.fill_(0) if module.bias.requires_grad else None
			except AttributeError:
				pass
		elif any([isinstance(module, cl) for cl in recurrent_classes]):
			for name, param in module.named_parameters():
				if name.startswith("weight"):
					nn.init.xavier_normal_(param.data) if param.requires_grad else None
				elif name.startswith("bias"):
					if param.requires_grad:
						hidden_size = param.size(0)
						param.data.fill_(0)
						param.data[hidden_size//4:hidden_size//2] = lstm_forget_bias_init

	@staticmethod
	def xavier_uniform(module, lstm_forget_bias_init=2):
		"""
		Xavier Uniform initialization.
		"""
		lstm_forget_bias_init = float(lstm_forget_bias_init) / 2
		normal_classes = (nn.Conv2d, nn.Linear, nn.Embedding)
		recurrent_classes = (nn.RNN, nn.LSTM, nn.GRU)
		if any([isinstance(module, cl) for cl in normal_classes]):
			nn.init.xavier_uniform_(module.weight.data) if module.weight.requires_grad else None
			try:
				module.bias.data.fill_(0) if module.bias.requires_grad else None
			except AttributeError:
				pass
		elif any([isinstance(module, cl) for cl in recurrent_classes]):
			for name, param in module.named_parameters():
				if name.startswith("weight"):
					nn.init.xavier_uniform_(param.data) if param.requires_grad else None
				elif name.startswith("bias"):
					if param.requires_grad:
						hidden_size = param.size(0)
						param.data.fill_(0)
						param.data[hidden_size//4:hidden_size//2] = lstm_forget_bias_init

	@staticmethod
	def orthogonal(module, lstm_forget_bias_init=2):
		"""
		Orthogonal initialization.
		"""
		lstm_forget_bias_init = float(lstm_forget_bias_init) / 2
		normal_classes = (nn.Conv2d, nn.Linear, nn.Embedding)
		recurrent_classes = (nn.RNN, nn.LSTM, nn.GRU)
		if any([isinstance(module, cl) for cl in normal_classes]):
			nn.init.orthogonal_(module.weight.data) if module.weight.requires_grad else None
			try:
				module.bias.data.fill_(0) if module.bias.requires_grad else None
			except AttributeError:
				pass
		elif any([isinstance(module, cl) for cl in recurrent_classes]):
			for name, param in module.named_parameters():
				if name.startswith("weight"):
					nn.init.orthogonal_(param.data) if param.requires_grad else None
				elif name.startswith("bias"):
					if param.requires_grad:
						hidden_size = param.size(0)
						param.data.fill_(0)
						param.data[hidden_size//4:hidden_size//2] = lstm_forget_bias_init


class Drawer(object):

	@staticmethod
	def put_text(img, text, coord=(10, 10), font_size=16):
		"""
		Put text to the image.
		--------------------
		Arguments:
			img (ndarray: H x W x C): image data.
			text (str): text to put.
			coord (tuple): coordination in the image.
			font_size (int): the size of text. 
		"""
		draw = ImageDraw.Draw(img)
		font = ImageFont.truetype("FreeSans.ttf", font_size)
		draw.text(coord, text, (255, 255, 255), font=font)
		
		return img

	@staticmethod
	def mask_img(img, attn, upscale=32):
		"""
		Put attention weights to each region in image.
		--------------------
		Arguments:
			img (ndarray: H x W x C): image data.
			attn (ndarray: 14 x 14): attention weights of each region.
			upscale (int): the ratio between attention size and image size.
		"""
		attn = transform.pyramid_expand(attn, upscale=upscale, sigma=20)
		attn = misc.toimage(attn).convert("L")
		mask = misc.toimage(np.zeros(img.shape, dtype=np.uint8)).convert("RGBA")
		img = misc.toimage(img).convert("RGBA")
		img = Image.composite(img, mask, attn)

		return img

	@staticmethod
	def mask_ques(sen, attn, idx2word):
		"""
		Put attention weights to each word in sentence.
		--------------------
		Arguments:
			sen (LongTensor): encoded sentence.
			attn (FloatTensor): attention weights of each word.
			idx2word (dict): vocabulary.
		"""
		fig, ax = plt.subplots(figsize=(15,15))
		ax.matshow(attn, cmap='bone')
		y = [1]
		x = [1] + [idx2word[i] for i in sen]
		ax.set_yticklabels(y)
		ax.set_xticklabels(x)
		ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
		ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

	@staticmethod
	def print_txt(sen, idx2word):
		"""
		Decode sentence and print it.
		--------------------
		Arguments:
			sen (LongTensor): encoded sentence.
			idx2word (dict): vocabulary.
		"""
		print(" ".join([idx2word[i] for i in sen]))

	@staticmethod
	def im_show(img):
		"""
		Show the image.
		--------------------
		Arguments:
			img (ndarray: H x W x C): image data.
		"""
		return misc.toimage(img).convert("RGBA")


class TimeMeter(object):
	"""Computes the average occurrence of some event per second"""
	def __init__(self, init=0):
		self.reset(init)

	def reset(self, init=0):
		self.init = init
		self.start = time.time()
		self.n = 0

	def update(self, val=1):
		self.n += val

	@property
	def avg(self):
		return self.n / self.elapsed_time
	
	@property
	def elapsed_time(self):
		return self.init + (time.time() - self.start)


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class StopwatchMeter(object):
	"""Computes the sum/avg duration of some event in seconds"""
	def __init__(self):
		self.reset()

	def start(self):
		self.start_time = time.time()

	def stop(self, n=1):
		if self.start_time is not None:
			delta = time.time() - self.start_time
			self.sum += delta
			self.n += n
			self.start_time = None

	def reset(self):
		self.sum = 0
		self.n = 0
		self.start_time = None

	@property
	def avg(self):
		return self.sum / self.n


def move_to_cuda(tensors, devices=None):
	if devices is not None:
		if len(devices) >= 1:
			cuda_tensors = []
			for tensor in tensors:
				if not isinstance(tensor, Sequence):
					cuda_tensors.append(tensor.cuda(devices[0], non_blocking=True))
				else:
					cuda_tensors.append(move_to_cuda(tensor, devices=devices))
			return tuple(cuda_tensors)
	return tensors


def save_checkpoint(model, state, is_best, is_save, directory):
	filename = os.path.join(directory, "{}.pth.tar".format(model))
	torch.save(state, filename)
	if is_best:
		filename_best = os.path.join(directory, "{}_best.pth.tar".format(model))
		shutil.copyfile(filename, filename_best)
	if is_save:
		filename_save = os.path.join(directory, "{}_epoch{}.pth.tar".format(model, state["last_epoch"]))
		shutil.copyfile(filename, filename_save)


def extract_statedict(model, excludes=None, is_parallel=False):
	excludes = [] if excludes is None else excludes
	state_dict = model.module.state_dict() if is_parallel else model.state_dict()
	model_state_dict = {}
	for k, v in state_dict.items():
		if not any([exclude in k for exclude in excludes]):
			model_state_dict[k] = v
	
	return model_state_dict
