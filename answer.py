
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import argparse
import json

from torch.autograd import Variable
from dense_coattn.data import Dataset, RCNN_Dataset, DataLoader
from dense_coattn.modules import LargeEmbedding
from dense_coattn.model import DCN, DCNWithAns, DCNWithRCNN, DCNWithRCNNAns


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


def answer(dataloader, model, idx2ans, opt, ensemble=False):
	"""
	Generate answers for testing the model.
	--------------------
	Arguments:
		dataloader: dataloader to provide data for the network.
		model: our trained network.
		idx2ans: set of possible answers.
		opt: testing option.
	"""
	model.eval()
	num_batches = len(dataloader)
	answers = []
	scores = []
	for i, batch in enumerate(dataloader):
		if not opt.use_rcnn:
			img, ques, ques_mask, ques_idx = batch
		else:
			img, ques, img_mask, ques_mask, ques_idx = batch

		img = Variable(img, volatile=True)
		img_mask = Variable(img_mask, volatile=True) if opt.use_rcnn else None
		ques = Variable(ques, volatile=True)
		ques_mask = Variable(ques_mask, volatile=True)

		img, img_mask, ques, ques_mask = move_to_cuda((img, img_mask, ques, ques_mask), devices=opt.gpus)
		ques = model.word_embedded(ques)

		score = model(img, ques, img_mask, ques_mask) if opt.use_rcnn else \
			model(img, ques, img_mask, ques_mask, is_train=False)
		_, inds = torch.sort(score, dim=1, descending=True)

		for j in range(ques_idx.size(0)):
			answers.append({"question_id": ques_idx[j], "answer": idx2ans[inds.data[j, 0]]})
		scores.append(score.data.cpu()) if ensemble else None
		if i % 10 == 0:
			print("processing %i / %i" % (i, num_batches))

	with open("%s.json" % (opt.save_file), "w") as file:
		json.dump(answers, file)

	if ensemble:
		scores = torch.cat(scores, dim=0)
		torch.save(scores, "%s.pt" % (opt.save_file))
	print("Done!")


def main(opt):
	"""
	Generating answers for (image, question) pair in the dataset.
	"""
	print("Constructing the dataset...")
	testset = Dataset(opt.data_path, opt.data_name, "test", opt.seq_per_img, opt.img_name,
		opt.size_scale, use_h5py=opt.use_h5py) if not opt.use_rcnn else \
			RCNN_Dataset(opt.data_path, opt.data_name, "test", opt.seq_per_img)
	testLoader = DataLoader(testset, batch_size=opt.batch, shuffle=False, 
		num_workers=opt.num_workers, pin_memory=True, drop_last=False, use_thread=opt.use_thread)

	idx2word = testset.idx2word
	idx2ans = testset.idx2ans
	ans_pool = testset.ans_pool
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

	dict_checkpoint = opt.train_from
	if dict_checkpoint:
		print("Loading model from checkpoint at %s" % dict_checkpoint)
		checkpoint = torch.load(dict_checkpoint)
		model.load_state_dict(checkpoint["model"])

	if len(opt.gpus) >= 1:
		model.cuda(opt.gpus[0])
	model.word_embedded = word_embedded

	print("Generating answers...")
	with torch.cuda.device(opt.gpus[0]):
		answer(testLoader, model, idx2ans, opt, ensemble=opt.ensemble)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--num_layers", type=int, default=1)
	parser.add_argument("--seq_per_img", type=int, default=1)
	parser.add_argument("--droprnn", type=float, default=0.1)
	parser.add_argument("--dropout", type=float, default=0.3)
	parser.add_argument("--dropattn", type=float, default=0)
	parser.add_argument("--cnn_name", type=str, default="resnet152")
	parser.add_argument("--hidden_size", type=int, default=1024)
	parser.add_argument("--wdim", type=int, default=256)
	parser.add_argument("--num_img_attn", type=int, default=1)
	parser.add_argument("--num_dense_attn", type=int, default=4)
	parser.add_argument("--num_predict_attn", type=int, default=4)
	parser.add_argument("--num_none", type=int, default=3)
	parser.add_argument("--num_seq", type=int, default=2)
	parser.add_argument("--predict_type", type=str, default="cat_attn")
	parser.add_argument("--gpus", type=int, nargs="+", default=[0])
	parser.add_argument("--data_path", type=str, default="/home/duykien/storage/vqa/dataset")
	parser.add_argument("--data_name", type=str, default="cocotrainval")
	parser.add_argument("--img_name", type=str, default="cocoimages")
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--batch", type=int, default=256)
	parser.add_argument("--word_vectors", type=str, default="/home/duykien/storage/vqa/dataset/glove_840B.pt")
	parser.add_argument("--train_from", default=None)
	parser.add_argument("--save_file", type=str, default="")
	parser.add_argument("--ensemble", action="store_true")
	parser.add_argument("--use_h5py", action="store_true")
	parser.add_argument("--use_rcnn", action="store_true")
	parser.add_argument("--use_thread", action="store_true")
	parser.add_argument("--size_scale", default=(448, 448))
	args = parser.parse_args()

	params = vars(args)
	print("Parsed input parameters:")
	print(json.dumps(params, indent=2))
	main(args)