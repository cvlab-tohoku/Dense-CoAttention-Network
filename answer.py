
import argparse
import json

import torch
from torch.utils.data import DataLoader

from dense_coattn.config import get_answer_config
from dense_coattn.data import BatchSampler, VQADataset, default_collate
from dense_coattn.model import DCN, DCNWithRCNN
from dense_coattn.modules import LargeEmbedding
from dense_coattn.utils import move_to_cuda


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
	dataset_length = len(dataloader.dataset)
	for i, batch in enumerate(dataloader):
		ques_idx = batch[-1]
		img_info, ques, ques_mask = move_to_cuda(batch[:-2], devices=opt.gpus)
		ques = model.word_embedded(ques)
		img, img_mask = img_info

		score = model(img, ques, img_mask, ques_mask)
		_, inds = torch.sort(score, dim=1, descending=True)

		for j in range(min(ques_idx.size(0), dataset_length - i*opt.batch_size)):
			answers.append({"question_id": ques_idx[j].item(), 
							"answer": idx2ans[inds[j, 0].item()]})
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
	testset = VQADataset(opt.data_path, opt.data_name, "test", opt.img_path, opt.img_type, "test")
	testLoader = DataLoader(testset, batch_size=opt.batch_size, shuffle=False, drop_last=False,
		num_workers=opt.num_workers, pin_memory=True, collate_fn=default_collate, batch_sampler=BatchSampler)

	idx2word = testset.idx2word
	idx2ans = testset.idx2ans

	print("Building model...")
	word_embedded = LargeEmbedding(len(idx2word), 300, padding_idx=0, devices=opt.gpus)
	word_embedded.load_pretrained_vectors(opt.word_vectors)

	num_ans = testset.ans_pool.shape[0]
	if opt.arch == "DCNWithRCNN":
		model = DCNWithRCNN(opt, num_ans)
	elif opt.arch == "DCN":
		model = DCN(opt, num_ans)

	dict_checkpoint = opt.resume
	if dict_checkpoint:
		print("Loading model from checkpoint at %s" % dict_checkpoint)
		checkpoint = torch.load(dict_checkpoint)
		model.load_state_dict(checkpoint["state_dict"])

	if len(opt.gpus) >= 1:
		model.cuda(opt.gpus[0])
	model.word_embedded = word_embedded

	print("Generating answers...")
	with torch.no_grad():
		answer(testLoader, model, idx2ans, opt, ensemble=opt.ensemble)


if __name__ == "__main__":
	args = get_answer_config()
	params = vars(args)
	print("Parsed input parameters:")
	print(json.dumps(params, indent=2))
	with torch.no_grad():
		main(args)
