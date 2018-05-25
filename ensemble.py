
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import argparse
import glob
import os
import random
import json

from torch.nn import functional as F
from dense_coattn.data import Dataset, RCNN_Dataset, DataLoader


def main(opt):
	"""
	Generating answers for (image, question) pair in the dataset.
	"""
	random.seed(opt.seed)
	print("Constructing the dataset...")
	testset = testset = Dataset(opt.data_path, opt.data_name, "test", opt.seq_per_img, opt.img_name,
		opt.size_scale, use_h5py=opt.use_h5py) if not opt.use_rcnn else \
			RCNN_Dataset(opt.data_path, opt.data_name, "test", opt.seq_per_img)
	testLoader = DataLoader(testset, batch_size=opt.batch, shuffle=False, 
		num_workers=opt.num_workers, pin_memory=True, drop_last=False, use_thread=opt.use_thread)

	idx2word = testset.idx2word
	idx2ans = testset.idx2ans
	num_batches = len(testLoader)

	file_name = glob.glob(os.path.join(opt.result_path, "DCN*.pt"))
	score = 0
	answers = []

	chosen_file = random.sample(file_name, opt.num_model)

	for name in chosen_file:
		score += F.sigmoid(torch.load(name))

	print("Number of results:", len(chosen_file))
	_, inds = torch.sort(score, dim=1, descending=True)

	for i, batch in enumerate(testLoader):
		ques_idx = batch[-1]

		for j in range(ques_idx.size(0)):
			answers.append({"question_id": ques_idx[j], "answer": idx2ans[inds[i*ques_idx.size(0) + j, 0]]})
		if i % 10 == 0:
			print("processing %i / %i" % (i, num_batches))

	with open("%s.json" % (os.path.join(opt.result_path, opt.save_file)), "w") as file:
		json.dump(answers, file)

	print(chosen_file)
	torch.save(chosen_file, "%s.pt" % (os.path.join(opt.result_path, opt.save_file)))

	print("Done!")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--num_workers", type=int, default=8)
	parser.add_argument("--batch", type=int, default=1000)
	parser.add_argument("--seq_per_img", type=int, default=1)
	parser.add_argument("--img_name", type=str, default="cocoimages")
	parser.add_argument("--data_path", type=str, default="/home/duykien/storage/vqa/dataset")
	parser.add_argument("--data_name", type=str, default="cocotrainval")
	parser.add_argument("--result_path", type=str, default="/home/duykien/storage/vqa/result")
	parser.add_argument("--num_model", type=int, default=9)
	parser.add_argument("--save_file", type=str, default="")
	parser.add_argument("--use_h5py", action="store_true")
	parser.add_argument("--use_thread", action="store_true")
	parser.add_argument("--use_rcnn", action="store_true")
	parser.add_argument("--size_scale", default=(448, 448))
	parser.add_argument("--seed", type=int, default=1234)
	args = parser.parse_args()

	params = vars(args)
	print("Parsed input parameters:")
	print(json.dumps(params, indent=2))
	main(args)
