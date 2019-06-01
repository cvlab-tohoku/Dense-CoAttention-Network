
import copy
import json
import sys

from dense_coattn.utils import TimeMeter

string_classes = (str, bytes)


class VQA(object):

	def __init__(self, annotation_file=None, question_file=None):
		self.questions = {}
		self.dataset = {}
		self.qa = {}
		self.qqa = {}
		self.img2qa = {}
		self.timer = TimeMeter()
		self.result = {}

		if not annotation_file == None and not question_file == None:
			print('Loading VQA annotations and questions into memory...')
			self.timer.reset()
			dataset = json.load(open(annotation_file, 'r'))
			questions = json.load(open(question_file, 'r'))
			print(self.timer.elapsed_time)
			self.dataset = dataset
			self.questions = questions
			self.create_index()

	def create_index(self):
		print('Creating index...')
		img2qa = {ann['image_id']: [] for ann in self.dataset['annotations']}
		qa = {ann['question_id']: [] for ann in self.dataset['annotations']}
		qqa = {ann['question_id']: [] for ann in self.dataset['annotations']}
		for ann in self.dataset['annotations']:
			img2qa[ann['image_id']] += [ann]
			qa[ann['question_id']] = ann
		for ques in self.questions['questions']:
			qqa[ques['question_id']] = ques
		print('Index created!')

		self.qa = qa
		self.qqa = qqa
		self.img2qa = img2qa

	def info(self):
		for key, value in self.dataset['info'].items():
			print('{}: {}'.format(key, value))

	def get_ques_ids(self, img_ids=[], ques_types=[], ans_types=[]):
		img_ids = img_ids if isinstance(img_ids, list) else [img_ids]
		ques_types = ques_types if isinstance(ques_types, list) else [ques_types]
		ans_types = ans_types if isinstance(ans_types, list) else [ans_types]

		if len(img_ids) == len(ques_types) == len(ans_types) == 0:
			anns = self.dataset['annotations']
		else:
			if not len(img_ids) == 0:
				anns = sum([self.img2qa[img_id] for img_id in img_ids if img_id in self.img2qa],[])
			else:
				anns = self.dataset['annotations']
			anns = anns if len(ques_types) == 0 else [ann for ann in anns if ann['question_type'] in ques_types]
			anns = anns if len(ans_types) == 0 else [ann for ann in anns if ann['answer_type'] in ans_types]
		ids = [ann['question_id'] for ann in anns]
		
		return ids

	def get_img_ids(self, ques_ids=[], ques_types=[], ans_types=[]):
		ques_ids = ques_ids if isinstance(ques_ids, list) else [ques_ids]
		ques_types = ques_types if isinstance(ques_types, list) else [ques_types]
		ans_types = ans_types if isinstance(ans_types, list) else [ans_types]

		if len(ques_ids) == len(ques_types) == len(ans_types) == 0:
			anns = self.dataset['annotations']
		else:
			if not len(ques_ids) == 0:
				anns = sum([self.qa[ques_id] for ques_id in ques_ids if ques_id in self.qa], [])
			else:
				anns = self.dataset['annotations']
			anns = anns if len(ques_types) == 0 else [ann for ann in anns if ann['question_type'] in ques_types]
			anns = anns if len(ans_types) == 0 else [ann for ann in anns if ann['answer_type'] in ans_types]
		ids = [ann['image_id'] for ann in anns]

		return ids

	def load_qa(self, ques_ids=[]):
		if isinstance(ques_ids, list):
			return [self.qa[ques_id] for ques_id in ques_ids]
		else:
			return [self.qa[ques_ids]]

	def show_qa(self, anns):
		if len(anns) == 0:
			return 0
		
		for ann in anns:
			ques_id = ann['question_id']
			print('Question: {}'.format(self.qqa[ques_id]['question']))
			for ans in ann['answers']:
				print('Answer {}: {}'.format(ans['answer_id'], ans['answer']))

	def load_result(self, result):
		preds = None
		if isinstance(result, list):
			preds = result
		elif isinstance(result, string_classes):
			preds = json.load(open(result))
		else:
			raise ValueError('Only path file or list of answers are accepted!')
		ques_ids = [res['question_id'] for res in preds]
		assert set(ques_ids) == set(self.get_ques_ids()), \
			'Results do not correspond to current VQA set.'
		for res in preds:
			self.result[res['question_id']] = res
