
import re
import sys


class VQAEval(object):

	def __init__(self, vqa, n=2):
		self.n = n
		self.accuracy = {}
		self.entropy = {}
		self.eval_qa = {}
		self.eval_questype = {}
		self.eval_anstype = {}
		self.vqa = vqa
		self.params = {'question_id': vqa.get_ques_ids()} if vqa is not None else None
		self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
							 "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
							 "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
							 "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
							 "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
							 "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
							 "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
							 "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
							 "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
							 "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
							 "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
							 "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
							 "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
							 "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
							 "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
							 "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
							 "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
							 "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
							 "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
							 "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
							 "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
							 "youll": "you'll", "youre": "you're", "youve": "you've"}
		self.manual_map    = { 'none': '0',
							  'zero': '0',
							  'one': '1',
							  'two': '2',
							  'three': '3',
							  'four': '4',
							  'five': '5',
							  'six': '6',
							  'seven': '7',
							  'eight': '8',
							  'nine': '9',
							  'ten': '10'
							}
		self.articles     = ['a',
							 'an',
							 'the'
							]

		self.period_strip  = re.compile("(?!<=\d)(\.)(?!\d)")
		self.comma_strip   = re.compile("(\d)(\,)(\d)")
		self.punct        = [';', r"/", '[', ']', '"', '{', '}',
							 '(', ')', '=', '+', '\\', '_', '-',
							 '>', '<', '@', '`', ',', '?', '!']

	def evaluate(self, ques_ids=None, use_progress_bar=False):
		if ques_ids is None:
			ques_ids = [ques_id for ques_id in self.params['question_id']]
		gts = {}
		res = {}
		for ques_id in ques_ids:
			gts[ques_id] = self.vqa.qa[ques_id]
			res[ques_id] = self.vqa.result[ques_id]

		acc_qa = []
		acc_questype = {}
		acc_anstype = {}
		print('Computing accuracy...')
		step = 0
		for ques_id in ques_ids:
			res_ans = res[ques_id]['answer']
			res_ans = res_ans.replace('\n', ' ')
			res_ans = res_ans.replace('\t', ' ')
			res_ans = res_ans.strip()
			res_ans = self.proccess_punctuation(res_ans)
			res_ans = self.process_digit_article(res_ans)
			gt_acc = []
			gt_answers = [ans['answer'] for ans in gts[ques_id]['answers']]
			if len(set(gt_answers)) > 1:
				for ans_dic in gts[ques_id]['answers']:
					ans_dic['answer'] = self.proccess_punctuation(ans_dic['answer'])
			for gt_ans_datum in gts[ques_id]['answers']:
				other_gt_ans = [item for item in gts[ques_id]['answers'] if item != gt_ans_datum]
				matching_ans = [item for item in other_gt_ans if item['answer'] == res_ans]
				acc = min(1, float(len(matching_ans))/3)
				gt_acc.append(acc)

			ques_type = gts[ques_id]['question_type']
			ans_type = gts[ques_id]['answer_type'] 
			avg_gt_acc = float(sum(gt_acc))/len(gt_acc)
			acc_qa.append(avg_gt_acc)
			if ques_type not in acc_questype:
				acc_questype[ques_type] = []
			acc_questype[ques_type].append(avg_gt_acc)
			if ans_type not in acc_anstype:
				acc_anstype[ans_type] = []
			acc_anstype[ans_type].append(avg_gt_acc)
			self.set_eval_qa(ques_id, avg_gt_acc)
			self.set_eval_questype(ques_id, ques_type, avg_gt_acc)
			self.set_eval_anstype(ques_id, ans_type, avg_gt_acc)
			if step % 100 == 0 and use_progress_bar:
				self.update_process(step/float(len(ques_ids)))
			step = step + 1
		
		self.set_accuracy(acc_qa, acc_questype, acc_anstype)
		print('Done computing accuracy...')

	def compute_entropy(self, ques_ids=None, use_progress_bar=False):
		if ques_ids is None:
			ques_ids = [ques_id for ques_id in self.params['question_id']]
		gts = {}
		res = {}
		for ques_id in ques_ids:
			gts[ques_id] = self.vqa.qa[ques_id]
			res[ques_id] = self.vqa.result[ques_id]

		ent_qa = []
		ent_questype = {}
		ent_anstype = {}
		print('Computing entropy...')
		step = 0
		for ques_id in ques_ids:
			ques_type = gts[ques_id]['question_type']
			ans_type = gts[ques_id]['answer_type']
			ent_qa.append(res[ques_id]['entropy'])
			if ques_type not in ent_questype:
				ent_questype[ques_type] = []
			ent_questype[ques_type].append(res[ques_id]['entropy'])
			if ans_type not in ent_anstype:
				ent_anstype[ans_type] = []
			ent_anstype[ans_type].append(res[ques_id]['entropy'])
			if step % 100 == 0 and use_progress_bar:
				self.update_process(step/float(len(ques_ids)))
			step = step + 1

		self.entropy['overall'] = sum(ent_qa)
		self.entropy['per_questype'] = {ques_type: sum(ent_questype[ques_type]) for ques_type in ent_questype}
		self.entropy['per_anstype'] = {ans_type: sum(ent_anstype[ans_type]) for ans_type in ent_anstype}

	def proccess_punctuation(self, intext):
		outtext = intext
		for p in self.punct:
			if (p + ' ' in intext or ' ' + p in intext) or (re.search(self.comma_strip, intext) != None):
				outtext = outtext.replace(p, '')
			else:
				outtext = outtext.replace(p, ' ')
		outtext = self.period_strip.sub("", outtext, re.UNICODE)

		return outtext

	def process_digit_article(self, intext):
		outtext = []
		temp = intext.lower().split()
		for word in temp:
			word = self.manual_map.setdefault(word, word)
			if word not in self.articles:
				outtext.append(word)
			else:
				pass
		
		for word_id, word in enumerate(outtext):
			if word in self.contractions:
				outtext[word_id] = self.contractions[word]
		outtext = ' '.join(outtext)

		return outtext

	def set_accuracy(self, acc_qa, acc_questype, acc_anstype):
		self.accuracy['overall'] = round(100*float(sum(acc_qa))/len(acc_qa), self.n)
		self.accuracy['per_questype'] = {ques_type: 
			round(100*float(sum(acc_questype[ques_type]))/len(acc_questype[ques_type]), self.n) 
				for ques_type in acc_questype}
		self.accuracy['per_anstype'] = {ans_type:
			round(100*float(sum(acc_anstype[ans_type]))/len(acc_anstype[ans_type]), self.n) 
				for ans_type in acc_anstype}

	def set_eval_qa(self, ques_id, acc):
		self.eval_qa[ques_id] = round(100*acc, self.n)

	def set_eval_questype(self, ques_id, ques_type, acc):
		if ques_type not in self.eval_questype:
			self.eval_questype[ques_type] = {}
		self.eval_questype[ques_type][ques_id] = round(100*acc, self.n)

	def set_eval_anstype(self, ques_id, ans_type, acc):
		if ans_type not in self.eval_anstype:
			self.eval_anstype[ans_type] = {}
		self.eval_anstype[ans_type][ques_id] = round(100*acc, self.n)

	def update_process(self, process):
		barLength = 20
		status = ""
		if isinstance(progress, int):
			progress = float(progress)
		if not isinstance(progress, float):
			progress = 0
			status = "error: progress var must be float\r\n"
		if progress < 0:
			progress = 0
			status = "Halt...\r\n"
		if progress >= 1:
			progress = 1
			status = "Done...\r\n"
		block = int(round(barLength*progress))
		text = "\rFinshed Percent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), int(progress*100), status)
		sys.stdout.write(text)
		sys.stdout.flush()
