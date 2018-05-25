
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import json
import argparse

from vqa_eval.vqaTools.vqa import VQA
from vqa_eval.vqaEval import VQAEval


def accuracy(taskType, dataType, dataSubType, resFile, dataDir, resultDir):
	annFile = "%s/v2_%s_%s_annotations.json" %(dataDir, dataType, dataSubType)
	quesFile = "%s/v2_%s_%s_%s_questions.json" %(dataDir, taskType, dataType, dataSubType)
	resultType = "real"
	fileType = "accuracy"
	fileName = resFile.split("/")[-1]

	accuracyFile = "%s/%s_%s_%s_%s_%s_%s" % (resultDir, taskType, dataType, dataSubType, resultType, fileType, fileName)

	vqa = VQA(annFile, quesFile)
	vqaRes = vqa.loadRes(resFile, quesFile)

	vqaEval = VQAEval(vqa, vqaRes, n=2)
	vqaEval.evaluate()

	print("\n")
	print("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy["overall"]))
	print("Per Question Type Accuracy is the following:")
	for quesType in vqaEval.accuracy["perQuestionType"]:
		print("%s : %.02f" %(quesType, vqaEval.accuracy["perQuestionType"][quesType]))
	print("\n")
	print("Per Answer Type Accuracy is the following:")
	for ansType in vqaEval.accuracy["perAnswerType"]:
		print("%s : %.02f" %(ansType, vqaEval.accuracy["perAnswerType"][ansType]))
	print("\n")

	json.dump(vqaEval.accuracy, open(accuracyFile, "w"))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--task_type", default="OpenEnded")
	parser.add_argument("--data_type", default="mscoco")
	parser.add_argument("--data_sub_type", default="val2014")
	parser.add_argument("--data_dir", default="/ceph/kien/data2.0")
	parser.add_argument("--result_file", default="/ceph/kien/VisualQA/analysis/ranknet8_19.json")
	parser.add_argument("--result_dir", default="/ceph/kien/VisualQA/result")

	args = parser.parse_args()
	params = vars(args)
	print('parsed input parameters:')
	print(json.dumps(params, indent=2))

	accuracy(args.task_type, args.data_type, args.data_sub_type, args.result_file, args.data_dir, args.result_dir)