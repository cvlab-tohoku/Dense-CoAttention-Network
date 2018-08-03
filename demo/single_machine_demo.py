
import random
import numpy as np
import os
import torch
import h5py
import argparse
import json
import torchvision.transforms as transforms
import sys
sys.path.insert(0, "../")

from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk

from torch.autograd import Variable
from dense_coattn.modules import LargeEmbedding
from dense_coattn.model import DCN, DCNWithAns, DCNWithRCNN, DCNWithRCNNAns

from nltk.tokenize import word_tokenize
import nltk
nltk.data.path.append("/ceph/kien/nltk_data")
UNK_WORD = "<unk>"


transform = transforms.Compose([
				transforms.Resize((448, 448)),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])


def move_to_cuda(tensors, devices=None):
	if devices is not None:
		if len(devices) >= 1:
			cuda_tensors = []
			for tensor in tensors:
				if tensor is not None:
					cuda_tensors.append(tensor.cuda(devices[0], async=True))
				else:
					cuda_tensors.append(None)
			return tuple(cuda_tensors)
	return tensors


def get_ques(sentence, word2idx, max_len_ques=14):
	processed_sen = word_tokenize(str(sentence).lower())
	final_ques = [w if w in word2idx else UNK_WORD for w in processed_sen]
	ques = torch.zeros(1, max_len_ques).long()

	for i, word in enumerate(final_ques):
		if i < max_len_ques:
			ques[0, i] = word2idx[word]
	ques_mask = ques.ne(0).float()

	return ques, ques_mask


def get_img(img_path):
	return transform(Image.open(img_path).convert("RGB")).unsqueeze(0)


def answer(sample, model, idx2ans, opt):
	"""
	Generate answers for testing the model.
	--------------------
	Arguments:
		dataloader: dataloader to provide data for the network.
		model: our trained network.
		idx2ans: set of possible answers.
		opt: testing option.
	"""
	img, ques, ques_mask = sample
	img = Variable(img, volatile=True)
	ques = Variable(ques, volatile=True)
	ques_mask = Variable(ques_mask, volatile=True)

	img, ques, ques_mask = move_to_cuda((img, ques, ques_mask), devices=opt.gpus)
	ques = model.word_embedded(ques)

	score = model(img, ques, None, ques_mask, is_train=False)
	_, inds = torch.sort(score, dim=1, descending=True)

	answer = [idx2ans[inds.data[0, i]] for i in range(opt.top_ans)]

	return answer


def load_pretrained_model(opt):
	"""
	Generating answers for (image, question) pair in the dataset.
	"""
	data_info = torch.load(os.path.join(opt.data_path, "%s_info.pt" % opt.data_name))
	word2idx = data_info["word2idx"]
	idx2word = data_info["idx2word"]
	idx2ans = data_info["idx2ans"]

	print("Building model...")
	word_embedded = LargeEmbedding(len(idx2word), 300, padding_idx=0, devices=opt.gpus)
	word_embedded.load_pretrained_vectors(opt.word_vectors)

	dict_checkpoint = opt.train_from
	if dict_checkpoint:
		print("Loading model from checkpoint at %s" % dict_checkpoint)
		model = torch.load(dict_checkpoint)

	if len(opt.gpus) >= 1:
		model.cuda(opt.gpus[0])
	model.word_embedded = word_embedded
	model.eval()

	return model, idx2ans, word2idx, opt


class Window(Frame):

	def __init__(self, master, model, idx2ans, word2idx, opt):
		Frame.__init__(self, master)
		self.master = master
		self.var = StringVar()
		self.img_path = None
		self.model = model
		self.idx2ans = idx2ans
		self.word2idx = word2idx
		self.opt = opt
		self.img_tensor = None
		self.answers = None
		self.img = None
		self.init_window()

	def init_window(self):
		self.master.title("DenseCoAttn demo!")
		self.pack(fill=BOTH, expand=1)

		menu = Menu(self.master)
		self.master.config(menu=menu)

		file = Menu(menu)
		file.add_command(label="Upload", command=self.showImg)
		file.add_command(label="Exit", command=self.client_exit)
		menu.add_cascade(label="File", menu=file)

		text = Label(self, text="Possible answers:")
		text.place(x=600, y=10)

		textbox = Entry(self.master, textvariable=self.var, width=70)
		textbox.focus_set()
		textbox.pack(pady=10, padx=10)

		button = Button(self.master, text="Answer", width=10, command=self.submitQues)
		button.pack()

	def showImg(self, max_size=500):
		img_path = filedialog.askopenfilename(initialdir = "./", title = "Select file")
		print(img_path)

		if img_path != "":
			self.img_tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)
			print(self.img_tensor.size())

			img_data = Image.open(img_path)
			width, height = img_data.size
			if width > height:
				height = height * max_size / width
				width = max_size
			else:
				width = width * max_size / height
				height = max_size
			width = int(width)
			height = int(height)
			render = ImageTk.PhotoImage(img_data.resize((width, height)))

			if self.img is not None:
				self.img.destroy()
			self.img = Label(self, image=render)
			self.img.image = render
			self.img.place(x=0, y=0)

	def submitQues(self):
		if self.answers is not None:
			self.answers.destroy()
		ques, ques_mask = get_ques(self.var.get(), self.word2idx)
		answers = answer((self.img_tensor, ques, ques_mask), self.model, self.idx2ans, self.opt)
		self.answers = Label(self, text="1. %s" % tuple(answers))
		self.answers.place(x=600, y=50)

	def client_exit(self):
		exit()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--top_ans", type=int, default=1)
	parser.add_argument("--gpus", type=int, nargs="+", default=[0])
	parser.add_argument("--data_path", type=str, default="dataset")
	parser.add_argument("--data_name", type=str, default="cocotrainval")
	parser.add_argument("--img_name", type=str, default="cocoimages")
	parser.add_argument("--word_vectors", type=str, default="dataset/glove_840B.pt")
	parser.add_argument("--train_from", default="model/pretrained_dcn.pt")
	args = parser.parse_args()

	params = vars(args)
	print("Parsed input parameters:")
	print(json.dumps(params, indent=2))
	model, idx2ans, word2idx, opt = load_pretrained_model(args)

	root = Tk()
	root.geometry("800x500")
	app = Window(root, model, idx2ans, word2idx, opt)
	root.mainloop()