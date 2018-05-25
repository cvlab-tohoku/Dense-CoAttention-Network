
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import argparse
import json
import sys
import torch.optim.lr_scheduler as lr_scheduler

from tensorboardX import SummaryWriter
from torch.autograd import Variable
from dense_coattn.model import DCN, DCNWithAns, DCNWithRCNN, DCNWithRCNNAns
from dense_coattn.modules import LargeEmbedding
from dense_coattn.data import Dataset, RCNN_Dataset, DataLoader
from dense_coattn.util import Initializer, Meter, Timer, Saver
from dense_coattn.optim import OptimWrapper, Adam, SGD, NoamOptimWrapper
from dense_coattn.evaluate import Accuracy
from dense_coattn.cost import BinaryLoss, LossCompute


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


def trainEpoch(epoch, dataloader, model, criterion, evaluation, optim, opt, writer):
	model.train()
	loss_record = [Meter() for _ in range(3)]
	accuracy_record = [Meter() for _ in range(3)]
	timer = Timer()

	timer.tic()
	optim.step_epoch()
	for i, batch in enumerate(dataloader):
		if not opt.use_rcnn:
			img, ques, ques_mask, _, ans_idx = batch
		else:
			img, ques, img_mask, ques_mask, _, ans_idx = batch

		img = Variable(img) if opt.use_rcnn else Variable(img, volatile=True)
		img_mask = Variable(img_mask) if opt.use_rcnn else None
		ques = Variable(ques, volatile=True)
		ques_mask = Variable(ques_mask)
		ans_idx = Variable(ans_idx)

		img, img_mask, ques, ques_mask, ans_idx = \
			move_to_cuda((img, img_mask, ques, ques_mask, ans_idx), devices=opt.gpus)
		ques = model.word_embedded(ques)
		ques = Variable(ques.data)

		optim.zero_grad()
		score = model(img, ques, img_mask, ques_mask) if opt.use_rcnn else \
			model(img, ques, img_mask, ques_mask, is_train=True)

		loss = criterion(score, ans_idx)
		loss.backward()
		accuracy = evaluation(Variable(score.data, volatile=True), Variable(ans_idx.data, volatile=True))
		_, ratio, updates, params = optim.step()

		for j in range(3):
			loss_record[j].update((loss.data[0] / opt.batch_size))
			accuracy_record[j].update(accuracy.data[0])

		if ratio is not None:
			writer.add_scalar("statistics/update_to_param_ratio", ratio, global_step=(epoch*len(dataloader) + i))
			writer.add_scalar("statistics/absolute_updates", updates, global_step=(epoch*len(dataloader) + i))
			writer.add_scalar("statistics/absolute_params", params, global_step=(epoch*len(dataloader) + i))
		
		if (i + 1) % 10 == 0:
			writer.add_scalar("iter/train_loss", loss_record[0].avg, global_step=(epoch*len(dataloader) + i))
			writer.add_scalar("iter/train_accuracy", accuracy_record[0].avg, global_step=(epoch*len(dataloader) + i))
			loss_record[0].reset()
			accuracy_record[0].reset()

		if (i + 1) % opt.log_interval == 0:
			print("Epoch %5d; iter %6i; loss: %8.2f; accuracy: %8.2f; %6.0fs elapsed" %
			  	(epoch, i+1, loss_record[1].avg, accuracy_record[1].avg, timer.toc(average=False)))
			loss_record[1].reset()
			accuracy_record[1].reset()
			timer.tic()

	writer.add_scalar("epoch/train_loss", loss_record[2].avg, global_step=epoch)
	writer.add_scalar("epoch/train_accuracy", accuracy_record[2].avg, global_step=epoch)

	return loss_record[2].avg, accuracy_record[2].avg


def evalEpoch(epoch, dataloader, model, criterion, evaluation, opt, writer):
	model.eval()
	total_loss, total_accuracy = Meter(), Meter()

	for batch in dataloader:
		if not opt.use_rcnn:
			img, ques, ques_mask, _, ans_idx = batch
		else:
			img, ques, img_mask, ques_mask, _, ans_idx = batch

		img = Variable(img, volatile=True)
		img_mask = Variable(img_mask, volatile=True) if opt.use_rcnn else None
		ques = Variable(ques, volatile=True)
		ques_mask = Variable(ques_mask, volatile=True)
		ans_idx = Variable(ans_idx, volatile=True)

		img, img_mask, ques, ques_mask, ans_idx = \
			move_to_cuda((img, img_mask, ques, ques_mask, ans_idx), devices=opt.gpus)
		ques = model.word_embedded(ques)

		score = model(img, ques, img_mask, ques_mask) if opt.use_rcnn else \
			model(img, ques, img_mask, ques_mask, is_train=False)
		accuracy = evaluation(score, ans_idx)
		loss = criterion(score, ans_idx)

		total_loss.update((loss.data[0] / opt.batch_size))
		total_accuracy.update(accuracy.data[0])

	writer.add_scalar("epoch/val_loss", total_loss.avg, global_step=epoch)
	writer.add_scalar("epoch/val_accuracy", total_accuracy.avg, global_step=epoch)

	return total_loss.avg, total_accuracy.avg


def trainModel(trainLoader, valLoader, model, criterion, evaluation, optim, opt):
	best_accuracy = None
	bad_counter = None
	history = None

	if valLoader is not None:
		best_accuracy = 0
		bad_counter = 0
		history = []
	writer = SummaryWriter(log_dir="logs/%s" % opt.save_model.split("/")[-1])

	for epoch in range(opt.num_epoch):
		print("----------------------------------------------")
		train_loss, train_accuracy = trainEpoch(epoch, trainLoader, model, criterion, 
			evaluation, optim, opt, writer)
		print("Train loss: %10.4f, accuracy: %5.2f" % (train_loss, train_accuracy))

		is_parallel = True if len(opt.gpus) > 1 else False
		model_state_dict = Saver.save_state_dict(model, excludes=["word_embedded"], is_parallel=is_parallel)
		Saver.save_model(model_state_dict, opt, epoch, best_accuracy, history, save_type=0)

		if valLoader is not None:
			val_loss, val_accuracy = evalEpoch(epoch, valLoader, model, criterion, evaluation, opt, writer)
			print("Val loss: %10.4f, accuracy: %5.2f" % (val_loss, val_accuracy))
			history.append(val_accuracy)

			if best_accuracy <= val_accuracy:
				best_accuracy = val_accuracy
				Saver.save_model(model_state_dict, opt, epoch, best_accuracy, history, save_type=1)
				bad_counter = 0

			if (len(history) > opt.patience) and val_accuracy <= torch.Tensor(history[:-opt.patience]).max():
				bad_counter += 1
				if bad_counter > opt.patience:
					print("Early Stop!")
					break

		if (epoch + 1) % opt.save_freq == 0:
			Saver.save_model(model_state_dict, opt, epoch, best_accuracy, history, save_type=2)
	writer.close()


def main(opt):
	Initializer.manual_seed(opt.seed)
	print("Constructing the dataset...")
	if opt.trainval == 0:
		trainset = Dataset(opt.data_path, opt.data_name, "train", opt.seq_per_img, opt.img_name, 
			opt.size_scale, use_h5py=opt.use_h5py) if not opt.use_rcnn else \
			RCNN_Dataset(opt.data_path, opt.data_name, "train", opt.seq_per_img)
		trainLoader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=opt.shuffle, 
			num_workers=opt.num_workers, pin_memory=opt.pin_memory, drop_last=opt.drop_last, use_thread=opt.use_thread)

		valset = Dataset(opt.data_path, opt.data_name, "val", opt.seq_per_img, opt.img_name, 
			opt.size_scale, use_h5py=opt.use_h5py) if not opt.use_rcnn else \
			RCNN_Dataset(opt.data_path, opt.data_name, "val", opt.seq_per_img)
		valLoader = DataLoader(valset, batch_size=opt.batch_size, shuffle=opt.shuffle,
			num_workers=opt.num_workers, pin_memory=opt.pin_memory, drop_last=opt.drop_last, use_thread=opt.use_thread)
	else:
		trainset = Dataset(opt.data_path, opt.data_name, "trainval", opt.seq_per_img, opt.img_name, 
			opt.size_scale, use_h5py=opt.use_h5py) if not opt.use_rcnn else \
			RCNN_Dataset(opt.data_path, opt.data_name, "trainval", opt.seq_per_img)
		trainLoader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=opt.shuffle, 
			num_workers=opt.num_workers, pin_memory=opt.pin_memory, drop_last=opt.drop_last, use_thread=opt.use_thread)

		valset = None
		valLoader = None

	idx2word = trainset.idx2word
	ans_pool = trainset.ans_pool
	ans_pool = torch.from_numpy(ans_pool)

	print("Building model...")
	word_embedded = LargeEmbedding(len(idx2word), 300, padding_idx=0, devices=opt.gpus)
	word_embedded.load_pretrained_vectors(opt.word_vectors)

	if opt.predict_type in ["sum_attn", "cat_attn", "prod_attn"]:
		num_ans = ans_pool.size(0)
		model = DCN(opt, num_ans) if not opt.use_rcnn else DCNWithRCNN(opt, num_ans)
	else:
		ans = word_embedded(Variable(ans_pool.cuda(opt.gpus[0]), volatile=True)).data
		ans_mask = ans_pool.ne(0).float()
		model = DCNWithAns(opt, ans, ans_mask) if not opt.use_rcnn else \
			DCNWithRCNNAns(opt, ans, ans_mask)

	criterion = BinaryLoss()
	evaluation = Accuracy()
	
	dict_checkpoint = opt.train_from
	if dict_checkpoint:
		print("Loading model from checkpoint at %s" % dict_checkpoint)
		checkpoint = torch.load(dict_checkpoint)
		model.load_state_dict(checkpoint["model"])

	if len(opt.gpus) >= 1:
		model.cuda(opt.gpus[0])

	if len(opt.gpus) > 1:
		model = nn.DataParallel(model, opt.gpus, dim=0)
	model.word_embedded = word_embedded

	optimizer = Adam(list(filter(lambda x: x.requires_grad, model.parameters())), lr=opt.lr,
		weight_decay=opt.weight_decay, record_step=opt.record_step)
	scheduler = lr_scheduler.StepLR(optimizer, opt.step_size, gamma=opt.gamma)
	optim_wrapper = OptimWrapper(optimizer, scheduler)

	nparams = []
	named_parameters = model.module.named_parameters() if len(opt.gpus) > 1 else model.named_parameters()
	for name, param in named_parameters:
		if not (name.startswith("resnet") or name.startswith("word_embedded") or name.startswith("ans")):
			nparams.append(param.numel())
	print("* Number of parameters: %d" % sum(nparams))

	checkpoint = None
	timer = Timer()
	timer.tic()
	try:
		with torch.cuda.device(opt.gpus[0]):
			trainModel(trainLoader, valLoader, model, criterion, evaluation, optim_wrapper, opt)
	except KeyboardInterrupt:
		print("It toke %.2f hours to train the network" % (timer.toc() / 3600))
		sys.exit("Training interrupted")

	print("It toke %.2f hours to train the network" % (timer.toc() / 3600))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--num_layers", type=int, default=2)
	parser.add_argument("--droprnn", type=float, default=0.1)
	parser.add_argument("--dropout", type=float, default=0.3)
	parser.add_argument("--dropattn", type=float, default=0)
	parser.add_argument("--seq_per_img", type=int, default=1)
	parser.add_argument("--cnn_name", type=str, default="resnet152")
	parser.add_argument("--save_model", type=str, default="/home/duykien/storage/vqa/model/DCN0")
	parser.add_argument("--hidden_size", type=int, default=1024)
	parser.add_argument("--wdim", type=int, default=256)
	parser.add_argument("--num_img_attn", type=int, default=4)
	parser.add_argument("--num_dense_attn", type=int, default=4)
	parser.add_argument("--num_predict_attn", type=int, default=4)
	parser.add_argument("--num_none", type=int, default=3)
	parser.add_argument("--num_seq", type=int, default=5)
	parser.add_argument("--predict_type", type=str, default="cat_attn")
	parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
	parser.add_argument("--log_interval", type=int, default=100)
	parser.add_argument("--record_step", type=int, default=1)
	parser.add_argument("--num_epoch", type=int, default=16)
	parser.add_argument("--patience", type=int, default=3)
	parser.add_argument("--save_freq", type=int, default=10)
	parser.add_argument("--trainval", type=int, default=0)
	parser.add_argument("--seed", type=int, default=12345)
	parser.add_argument("--data_path", type=str, default="/home/duykien/storage/vqa/dataset")
	parser.add_argument("--data_name", type=str, default="cocotrainval")
	parser.add_argument("--img_name", type=str, default="cocoimages")
	parser.add_argument("--num_workers", type=int, default=8)
	parser.add_argument("--batch_size", type=int, default=320)
	parser.add_argument("--word_vectors", type=str, default="/home/duykien/storage/vqa/dataset/glove_840B.pt")
	parser.add_argument("--lr", type=float, default=0.001)
	parser.add_argument("--gamma", type=float, default=0.5)
	parser.add_argument("--step_size", type=int, default=7)
	parser.add_argument("--weight_decay", type=float, default=0.0001)
	parser.add_argument("--size_scale", default=(448, 448))
	parser.add_argument("--train_from", default=None)
	parser.add_argument("--max_grad_norm", default=None)
	parser.add_argument("--use_h5py", action="store_true")
	parser.add_argument("--shuffle", action="store_true")
	parser.add_argument("--pin_memory", action="store_true")
	parser.add_argument("--drop_last", action="store_true")
	parser.add_argument("--use_thread", action="store_true")
	parser.add_argument("--use_rcnn", action="store_true")
	args = parser.parse_args()

	params = vars(args)
	print("Parsed input parameters:")
	print(json.dumps(params, indent=2))
	main(args)