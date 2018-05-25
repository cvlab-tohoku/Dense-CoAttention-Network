
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import json
import h5py
import numpy as np
import re
import sys
import os
import torch
import argparse

from collections import Counter


# Constants in the vocabulary
UNK_WORD = "<unk>"
PAD_WORD = "<_>"
PAD = 0


data_map_vqa = {
	"train": "mscoco_train.json",
	"val": "mscoco_val.json",
	"trainval": "mscoco_trainval.json",
	"testdev": "mscoco_testdev.json",
	"test": "mscoco_test.json",
	"train_comp_path": "/ceph/kien/data2.0/v2_mscoco_train2014_complementary_pairs.json",
	"val_comp_path": "/ceph/kien/data2.0/v2_mscoco_val2014_complementary_pairs.json",
}


def get_top_answers(examples, occurs=0):
	"""
	Extract all of correct answers in the dataset. Build a set of possible answers which
	appear more than pre-defined "occurs" times.
	--------------------
	Arguments:
		examples (list): the json data loaded from disk.
		occurs (int): a threshold that determine which answers are kept.
	Return:
		vocab_ans (list): a set of correct answers in the dataset.
	"""
	counter = Counter()
	for ex in examples:
		for ans in ex["mc_ans"]:
			ans = str(ans).lower()
			counter.update([ans])

	frequent_answers = list(filter(lambda x: x[1] > occurs, counter.items()))
	total_ans = sum(item[1] for item in counter.items())
	total_freq_ans = sum(item[1] for item in frequent_answers)

	print("Number of unique answers:", len(counter))
	print("Total number of answers:", total_ans)
	print("Top %i answers account for %f%%" % (len(frequent_answers), total_freq_ans*100.0/total_ans))
	print("Sample frequent answers:")
	print("\n".join(map(str, frequent_answers[:20])))

	vocab_ans = []
	for item in frequent_answers:
		vocab_ans.append(item[0])

	return vocab_ans


def filter_answers(examples, ans2idx):
	"""
	Remove the answers that don't appear in our answer set.
	--------------------
	Arguments:
		examples (list): the json data that contains all of answers in the dataset.
		ans2idx (dict): a set of considered answers.
	Return:
		examples (list): the processed json data which contains only answers in the answer set.
	"""
	for ex in examples:
		ex["ans"] = [list(filter(lambda x: x[0] in ans2idx, answers)) for answers in ex["ans"]]

	return examples


def tokenize(sentence):
	"""
	Normal tokenize implementation.
	--------------------
	Arguments:
		sentence (str): a setence that will be tokenized.
	Return:
		A list of tokens from the sentence.
	"""
	return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) \
		if i != "" and i != " " and i != "\n"]


def tokenize_mcb(sentence):
	"""
	MCB tokenize implementation.
	--------------------
	Arguments:
		sentence (str): a setence that will be tokenized.
	Return:
		A list of tokens from the sentence.
	"""
	for i in [r"\?", r"\!", r"\'", r"\"", r"\$", r"\:", r"\@", r"\(", r"\)", r"\,", r"\.", r"\;"]:
		sen = re.sub(i, "", sen)

	for i in [r"\-", r"\/"]:
		sen = re.sub(i, " ", sen)
	q_list = re.sub(r"\?", "", sen.lower()).split()
	q_list = list(filter(lambda x: len(x) > 0, q_list))

	return q_list


def process_text(examples, without_ans=False, nlp="nltk"):
	"""
	Create "processed_ques" and "processed_ans" where each question or answer is replaced
	by an array of processed tokens using tokenizer.
	--------------------
	Arguments:
		examples (list): the json data contains string of questions and answers.
		without_ans (bool): If True, the dataset doesn't contain answers.
		nlp (str): type of tokenize tool.
	Return:
		examples (list): the json data contains "processed_ques" and "processed_ans" fields.
	"""
	if nlp == "nltk":
		from nltk.tokenize import word_tokenize
		import nltk
		nltk.data.path.append("/ceph/kien/nltk_data")
		tokenizer = word_tokenize
	elif nlp == "mcb":
		tokenizer = tokenize_mcb
	else:
		tokenizer = tokenize

	print("Tokenizing questions and answers...")
	for i, ex in enumerate(examples):
		ex["processed_ques"] = [tokenizer(str(ques).lower()) for ques in ex["ques"]]
		ex["processed_ans"] = [list(map(lambda x: (tokenizer(str(x[0]).lower()), x[1]), answers)) \
			for answers in ex["ans"]] if not without_ans else None

		if i < 5:
			print(ex["processed_ques"])
			print(ex["processed_ans"]) if not without_ans else None

		if (i+1) % 10000 == 0:
			sys.stdout.write("processing %d/%d (%.2f%% done)	\r" %((i+1), len(examples), (i+1)*100.0/len(examples)))
			sys.stdout.flush()

	return examples


def process_ans(ans2idx, word2idx, max_len_ans, nlp="nltk"):
	"""
	Given the set of possible answers to predict, the function tokenize these answers and 
	replace each word with a corresponding index in the word2idx dictionary.
	--------------------
	Arguments:
		ans2idx (dict): a dictionary contains answers and its index.
		word2idx (dict): a dictionary contains words and its index.
		max_len_ans (int): a threshold that contrains the maximum length of possible answers.
		nlp (str): type of tokenize tool.
	Return:
		encoded_poss_ans (ndarray: num_ans x max_len_ans): a numpy array of possible answers 
			where each row is an answer and each column is a word.
	"""
	if nlp == "nltk":
		from nltk.tokenize import word_tokenize
		import nltk
		nltk.data.path.append("/ceph/kien/nltk_data")
		tokenizer = word_tokenize
	elif nlp == "mcb":
		tokenizer = tokenize_mcb
	else:
		tokenizer = tokenize

	possible_answers = [[word2idx[w] if w in word2idx else word2idx[UNK_WORD] for w in tokenizer(ans)] \
		for ans in ans2idx.keys()]
	encoded_poss_ans = np.zeros((len(possible_answers), max_len_ans), dtype=np.int64)
	for i, ans in enumerate(possible_answers):
		for j, w in enumerate(ans):
			if j < max_len_ans:
				encoded_poss_ans[i, j] = w

	return encoded_poss_ans


def build_glove_train(examples, gloves):
	"""
	Using a pre-defined vocabulary from GloVe. Convert all of word not being in the GloVe vocabulary
	to unk word and save the new questions and answers to "final_question", and "final_ans".
	--------------------
	Arguments:
		examples (list): the json data contains list of tokens for questions and answers.
		gloves (dict): total of GloVe words.
	Return:
		examples (list): the json data that filtered by GloVe vocab.
		max_len_ans (int): maximum length of answers in dataset.
		max_len_ques (int): maximum lenght of questions in dataset.
	"""
	counts = Counter()
	for ex in examples:
		for ques in ex["processed_ques"]:
			counts.update(ques)
		for answers in ex["processed_ans"]:
			for ans in answers:
				counts.update(ans)

	sorted_counts = sorted([(count, word) for word, count in counts.items()], reverse=True)
	print("Most frequent words in the dataset:")
	print("\n".join(map(str, sorted_counts[:20])))

	total_words = sum(counts.values())
	print("Total number of words:", total_words)
	print("Number of unique words in dataset:", len(counts))
	print("Number of words in GloVe:", len(gloves))

	words_diff = frozenset(counts.keys()).difference(frozenset(gloves.keys()))
	print("Number of unique words in unk: %d/%d = %.2f%%" 
		% (len(words_diff), len(counts), len(words_diff)*100./len(counts)))
	total_unk = sum(counts[word] for word in words_diff)
	print("Total number of unk words: %d/%d = %.2f%%" 
		% (total_unk, total_words, total_unk*100./total_words))

	# Check the length distribution of questions and answers (if possible)
	ques_lengths = Counter()
	ans_lengths = Counter()

	for ex in examples:
		for ques in ex["processed_ques"]:
			ques_lengths.update([len(ques)])
		for answers in ex["processed_ans"]:
			for ans in answers:
				ans_lengths.update([len(ans[0])])

	max_len_ques = max(ques_lengths.keys())
	max_len_ans = max(ans_lengths.keys())

	print("Max length question:", max_len_ques)
	print("Length distribution of questions (length, count):")
	total_questions = sum(ques_lengths.values())
	for i in range(max_len_ques+1):
		print("%2d: %10d \t %f%%" % (i, ques_lengths.get(i, 0), 
			ques_lengths.get(i, 0)*100./total_questions))

	print("Max length answer:", max_len_ans)
	print("Length distribution of answers (length, count):")
	total_answers = sum(ans_lengths.values())
	for i in range(max_len_ans+1):
		print("%2d: %10d \t %f%%" % (i, ans_lengths.get(i, 0), 
			ans_lengths.get(i, 0)*100./total_answers))

	for ex in examples:
		ex["final_ques"] = [[w if w in gloves else UNK_WORD for w in ques] \
			for ques in ex["processed_ques"]]
		ex["final_ans"] = [[(list(map(lambda w: w if w in gloves else UNK_WORD, ans[0])), ans[1]) \
			for ans in answers] for answers in ex["processed_ans"]]

	return examples, max_len_ques, max_len_ans


def filter_unk_word(examples, word2idx, without_ans=False):
	"""
	Given the constructed vocabulary from train or (train+val) set, convert all of words
	that don't appear in the vocabulary to unk.
	--------------------
	Arguments:
		examples (list): the json data of test set.
		word2idx (dict): the dictionary of vocabulary constructed using train or (train+val) dataset.
		without_ans (bool): If True, the dataset doesn't contain answers.
	Return:
		examples (list): the updated json data where words not being in the vocabulary are set to unk.
	"""
	for ex in examples:
		ex["final_ques"] = [[w if w in word2idx else UNK_WORD for w in ques]
			for ques in ex["processed_ques"]]
		ex["final_ans"] = [[(list(map(lambda w: w if w in word2idx else UNK_WORD, ans[0])), ans[1]) \
			for ans in answers] for answers in ex["processed_ans"]] if not without_ans else None

	return examples


def encode_ans(examples, ans2idx):
	"""
	Convert answers for each question to its index.
	--------------------
	Arguments:
		examples (list): the json data contains answers for each question.
		ans2idx (dict): dictionary of answers and its indices.
	Return:
		examples (list): the updated data where answers are replaced by its index.
	"""
	for ex in examples:
		ex["ans_id"] = [list(map(lambda x: (ans2idx[x[0]], x[1]), answers)) for answers in ex["ans"]]

	return examples


def encode_VQA(examples, max_len_ques, num_ans, word2idx, without_ans=False):
	"""
	Using the processed json data to create numpy array which contains information of
	questions, images, and index of correct answers in set of possible answers.
	--------------------
	Arguments:
		examples (list): the process json data.
		max_len_ques (int): the maximum length of question allowed.
		num_ans (int): number of possible answers in the pre-defined set.
		word2idx (dict): dictionary of vocabulary.
		without_ans (bool): If True, the dataset doesn't contain answers.
	Return:
		img_idx (ndarray: num_sample): index of images.
		ques_array (ndarray: num_ques x max_len_ques): question data in numpy array.
		txt_start_idx (ndarray: num_sample): start index of questions of the same image.
		txt_end_idx (ndarray: num_sample): end index of questions of the same image.
		ans_idx (ndarray: num_ques x num_poss_ans): ground truth scores of possible answers
		ques_idx (ndarray: num_ques): question index data corresponds to each question.
	"""
	N = len(examples)
	M = sum(len(ex["final_ques"]) for ex in examples)

	ques_array = np.zeros((M, max_len_ques), dtype=np.int64)
	img_idx = np.zeros(N, dtype=np.int64)
	txt_start_idx = np.zeros(N, dtype=np.int64)
	txt_end_idx = np.zeros(N, dtype=np.int64)
	ques_idx = np.zeros(M, dtype=np.int64)
	ans_idx = np.zeros((M, num_ans), dtype=np.float32) if not without_ans else None

	txt_counter = 0
	counter = 0

	for i, ex in enumerate(examples):
		n = len(ex["final_ques"])
		assert n > 0, "Some images has no questions"

		img_idx[i] = ex["id"]
		for j, ques in enumerate(ex["final_ques"]):
			ques_idx[txt_counter] = ex["ques_id"][j]

			if not without_ans:
				for ans in ex["ans_id"][j]:
					ans_idx[txt_counter, ans[0]] = ans[1]

			assert len(ques) > 0, "Question has no words!"
			for k, w in enumerate(ques):
				if k < max_len_ques:
					ques_array[txt_counter, k] = word2idx[w]

			txt_counter += 1

		txt_start_idx[i] = counter
		txt_end_idx[i] = counter + n - 1
		counter += n

	assert txt_counter == M, "Number of questions doesn't match!"
	print("Encoded array of questions:", str(ques_array.shape))

	return (img_idx, ques_array, txt_start_idx, txt_end_idx, ans_idx, ques_idx)


def process_dataset(dataset, num_occurs, glove_path, max_ques, max_ans):
	"""
	Process the loaded json file into a dataset which can be fed into a neural network.
	--------------------
	Arguments:
		dataset (list): the json data loaded from disk.
		num_occurs (int): a threshold that determine which answers are kept.
		glove_path (str): path points to the file storing GloVe vectors.
		max_ques (int): maximum length of question to be processed.
		max_ans (int): maximum length of answer to be processed.
	Return:
		ans2idx (dict): indices to possible answers.
		idx2ans (dict): possible answers to its indices.
		word2idx (dict): dictionary of vocabulary from words to indices.
		idx2word (dict): dictionary of vocabulary from indices to words.
		dataset (list): processed dataset which contains encoded information.
		max_len_ques (int): maximum length of questions in dataset if max_ques is not set.
		poss_answers (ndarray: num_ques x num_poss_ans): a set of answers to predict.
	"""
	top_answers = get_top_answers(dataset, num_occurs)
	num_ans = len(top_answers)
	ans2idx = {}
	for idx, ans in enumerate(top_answers):
		ans2idx[ans] = idx
	idx2ans = top_answers

	dataset = filter_answers(dataset, ans2idx)
	dataset = process_text(dataset)

	assert glove_path is not None, "Couldn't find GloVe file!"
	gloves = torch.load(glove_path)
	print("gloves type:", type(gloves))
	dataset, max_len_ques, max_len_ans = build_glove_train(dataset, gloves["word2idx"])
	idx2word = gloves["idx2word"]
	word2idx = gloves["word2idx"]

	max_len_ques = max_ques if max_ques is not None else max_len_ques
	max_len_ans = max_ans if max_ans is not None else max_len_ans

	dataset = encode_ans(dataset, ans2idx)
	poss_answers = process_ans(ans2idx, word2idx, max_len_ans)

	return ans2idx, idx2ans, word2idx, idx2word, dataset, max_len_ques, poss_answers


def create_dataset(data_path, data_name, data_type, dataset, max_len_ques, 
				   poss_ans, word2idx, comp=None):
	"""
	Create a h5py file and store all of numpy arrays of the dataset into that file.
	--------------------
	Arguments:
		data_path (str): path to the directory for storing the dataset file.
		data_name (str): name of the dataset.
		data_type (str): "train", "val", "trainval", "test", or "testdev".
		dataset list): the json data.
		max_len_ques (int): the maximum length of questions.
		poss_ans (ndarray: num_ques x num_poss_ans): a set of answers to predict.
		word2idx (dict): dictionary of vocabulary.
		comp (list): the json data containing complementary pairs.
	Return:
		Create the data set in h5py file.
	"""
	num_ans = poss_ans.shape[0]
	without_ans = True if data_type in ["testdev", "test"] else False
	(img_idx, ques_array, txt_start_idx, txt_end_idx, ans_idx, ques_idx) = \
		encode_VQA(dataset, max_len_ques, num_ans, word2idx, without_ans)

	if comp is not None:
		comp_idx = np.zeros((len(comp), 2), dtype=np.int64)
		for i, pair in enumerate(comp):
			comp_idx[i, 0] = np.argwhere(ques_idx == pair[0])[0, 0]
			comp_idx[i, 1] = np.argwhere(ques_idx == pair[1])[0, 0]

	file = h5py.File(os.path.join(data_path, "%s_%s.h5" % (data_name, data_type)), "w")
	file.create_dataset("img_idx", dtype=np.int64, data=img_idx)
	file.create_dataset("questions", dtype=np.int64, data=ques_array)
	file.create_dataset("txt_start_idx", dtype=np.int64, data=txt_start_idx)
	file.create_dataset("txt_end_idx", dtype=np.int64, data=txt_end_idx)
	file.create_dataset("ques_idx", dtype=np.int64, data=ques_idx)
	file.create_dataset("ans_pool", dtype=np.int64, data=poss_ans)
	file.create_dataset("ans_idx", dtype=np.float32, data=ans_idx) if not without_ans else None
	file.create_dataset("comp_idx", dtype=np.int64, data=comp_idx) if comp is not None else None
	file.close()


def main(opt):
	"""
	Create dataset for "train", "val", "test", and "testdev" or "trainval", "test", and "testdev".
	"""
	if not opt.trainval:
		print("Process train dataset...")
		train_set = json.load(open(os.path.join(opt.data_path, data_map_vqa["train"]), "r"))
		comp_train = json.load(open(data_map_vqa["train_comp_path"], "r"))

		ans2idx, idx2ans, word2idx, idx2word, train_set, max_len_ques, poss_answers = \
			process_dataset(train_set, opt.num_occurs, opt.glove_path, opt.max_ques, opt.max_ans)

		create_dataset(opt.data_path, opt.data_name, "train", train_set, max_len_ques, 
			poss_answers, word2idx, comp=comp_train)

		print("Process val dataset...")
		val_set = json.load(open(os.path.join(opt.data_path, data_map_vqa["val"]), "r"))
		comp_val = json.load(open(data_map_vqa["val_comp_path"], "r"))

		val_set = filter_answers(val_set, ans2idx)
		val_set = process_text(val_set)
		val_set = filter_unk_word(val_set, word2idx)
		val_set = encode_ans(val_set, ans2idx)

		create_dataset(opt.data_path, opt.data_name, "val", val_set, max_len_ques,
			poss_answers, word2idx, comp=comp_val)
	else:
		print("Process trainval dataset...")
		trainval_set = json.load(open(os.path.join(opt.data_path, data_map_vqa["trainval"]), "r"))
		comp_trainval = json.load(open(data_map_vqa["train_comp_path"], "r"))
		comp_trainval.extend(json.load(open(data_map_vqa["val_comp_path"], "r")))

		ans2idx, idx2ans, word2idx, idx2word, trainval_set, max_len_ques, poss_answers = \
			process_dataset(trainval_set, opt.num_occurs, opt.glove_path, opt.max_ques, opt.max_ans)

		create_dataset(opt.data_path, opt.data_name, "trainval", trainval_set, max_len_ques,
			poss_answers, word2idx, comp=comp_trainval)

	print("Process testdev dataset...")
	testdev_set = json.load(open(os.path.join(opt.data_path, data_map_vqa["testdev"]), "r"))

	testdev_set = process_text(testdev_set, without_ans=True)
	testdev_set = filter_unk_word(testdev_set, word2idx, without_ans=True)

	create_dataset(opt.data_path, opt.data_name, "testdev", testdev_set, max_len_ques,
		poss_answers, word2idx)

	print("Process test dataset...")
	test_set = json.load(open(os.path.join(opt.data_path, data_map_vqa["test"]), "r"))

	test_set = process_text(test_set, without_ans=True)
	test_set = filter_unk_word(test_set, word2idx, without_ans=True)

	create_dataset(opt.data_path, opt.data_name, "test", test_set, max_len_ques,
		poss_answers, word2idx)

	print("Saving information file...")
	info = {
		"ans2idx": ans2idx,
		"idx2ans": idx2ans,
		"word2idx": word2idx,
		"idx2word": idx2word,
	}
	torch.save(info, os.path.join(opt.data_path, "%s_info.pt" % (opt.data_name)))
	print("Done!")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", type=str, default="/ceph/kien/VQA/dataset")
	parser.add_argument("--data_name", type=str, default="")
	parser.add_argument("--max_ques", type=int, default=14)
	parser.add_argument("--max_ans", type=int, default=None)
	parser.add_argument("--glove_path", type=str, default="/ceph/kien/VQA/dataset/glove_840B.pt")
	parser.add_argument("--num_occurs", type=int, default=8)
	parser.add_argument("--trainval", action="store_true")

	args = parser.parse_args()
	params = vars(args)
	print("Parsed input parameters:")
	print(json.dumps(params, indent=2))
	main(args)