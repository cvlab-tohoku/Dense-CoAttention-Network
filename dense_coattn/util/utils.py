
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import matplotlib
matplotlib.use('Agg')
import time
import torch
import random
import numpy as np
import scipy.misc as misc
import skimage.transform as transform
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.nn as nn

from PIL import ImageDraw, ImageFont, Image


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
			nn.init.xavier_normal(module.weight.data) if module.weight.requires_grad else None
			try:
				module.bias.data.fill_(0) if module.bias.requires_grad else None
			except AttributeError:
				pass
		elif any([isinstance(module, cl) for cl in recurrent_classes]):
			for name, param in module.named_parameters():
				if name.startswith("weight"):
					nn.init.xavier_normal(param.data) if param.requires_grad else None
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
			nn.init.xavier_uniform(module.weight.data) if module.weight.requires_grad else None
			try:
				module.bias.data.fill_(0) if module.bias.requires_grad else None
			except AttributeError:
				pass
		elif any([isinstance(module, cl) for cl in recurrent_classes]):
			for name, param in module.named_parameters():
				if name.startswith("weight"):
					nn.init.xavier_uniform(param.data) if param.requires_grad else None
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
			nn.init.orthogonal(module.weight.data) if module.weight.requires_grad else None
			try:
				module.bias.data.fill_(0) if module.bias.requires_grad else None
			except AttributeError:
				pass
		elif any([isinstance(module, cl) for cl in recurrent_classes]):
			for name, param in module.named_parameters():
				if name.startswith("weight"):
					nn.init.orthogonal(param.data) if param.requires_grad else None
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


class Saver(object):

	@staticmethod
	def save_model(model_state_dict, opt, epoch, best_accuracy, history, save_type=0):
		"""
		Save our network.
		--------------------
		Arguments:
			model_state_dict (dict): contains modules and its parameters.
			opt (args object): option for the training procedure.
			epoch (int): checkpoint of epoch.
			best_accuracy (float): the best accuracy at the saving time.
			history (list): previous accuracies.
			save_type (int): "back up", "save best", or "save epoch".
		"""
		checkpoint = {
			"model": model_state_dict,
			"opt": opt,
			"epoch": epoch,
			"best_accuracy": best_accuracy,
			"history": history,
		}
		if save_type == 0:
			print("Backing up model...")
			model_name = "%s.pt" % opt.save_model
		elif save_type == 1:
			print("Saving the best model...")
			model_name = "%s_best.pt" % opt.save_model
		elif save_type == 2:
			print("Saving model at epoch %i..." % epoch)
			model_name = "%s_%s.pt" % (opt.save_model, epoch)
		else:
			raise TypeError("Invalid save type!")
		torch.save(checkpoint, model_name)

	@staticmethod
	def save_state_dict(model, excludes=None, is_parallel=False):
		"""
		Return the state dict of our network.
		--------------------
		Arguments:
			model (nn.Module): the trained network.
			excludes (list): modules which are not saved.
			is_parallel (bool): If True, the model is on multiple GPUs.
		Return:
			model_state_dict (dict): dictionary which stores modules' name and weights.
		"""
		excludes = [] if excludes is None else excludes
		state_dict = model.module.state_dict() if is_parallel else model.state_dict()
		model_state_dict = {}
		for k, v in state_dict.items():
			if not any([exclude in k for exclude in excludes]):
				model_state_dict[k] = v

		return model_state_dict


class Timer(object):

	def __init__(self):
		self.total_time = 0.
		self.calls = 0
		self.start_time = 0.
		self.diff = 0.
		self.average_time = 0.

	def tic(self):
		self.start_time = time.time()

	def toc(self, average=True):
		self.diff = time.time() - self.start_time
		self.total_time += self.diff
		self.calls += 1
		self.average_time = self.total_time / self.calls

		if average:
			return self.average_time
		return self.diff


class Meter(object):

	def __init__(self):
		self.reset()

	def reset(self):
		self.value = 0.
		self.avg = 0.
		self.sum = 0.
		self.count = 0

	def update(self, value, n=1):
		self.value = value
		self.sum += value * n
		self.count += n
		self.avg = self.sum / self.count