
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import glob
import re
import sys
import cv2
import h5py
import torch
import numpy as np
import argparse
import json

from threading import Thread, Lock
if sys.version_info[0] == 2:
	import Queue as queue
else:
	import queue


folder_map = {
	"train": ["train2014"],
	"val": ["val2014"],
	"trainval": ["train2014", "val2014"],
	"test": ["test2015"],
}


def save_images(image_path, image_type, data_path, data_name, num_workers):
	"""
	Process all of the image to a numpy array, then store them to a file.
	--------------------
	Arguments:
		image_path (str): path points to images.
		image_type (str): "train", "val", "trainval", or "test".
		data_path (str): path points to the location which stores images.
		data_name (str): name of stored file.
		num_workers (int): number of threads used to load images.
	"""
	dataset = h5py.File(os.path.join(data_path, "%s_%s.h5" % (data_name, image_type)), "w")

	q = queue.Queue()
	images_idx = {}
	images_path = []
	lock = Lock()
	for data in folder_map[image_type]:
		folder = os.path.join(image_path, data)
		images_path.extend(glob.glob(folder+"/*"))
	pattern = re.compile(r"_([0-9]+).jpg")

	for i, img_path in enumerate(images_path):
		assert len(pattern.findall(img_path)) == 1, "More than one index found in an image path!"
		idx = int(pattern.findall(img_path)[0])
		images_idx[idx] = i
		q.put((i, img_path))
	assert len(images_idx) == len(images_path), "Duplicated indices are found!"
	images = dataset.create_dataset("images", (len(images_path), 448, 448, 3), dtype=np.uint8)

	def _worker():
		while True:
			i, img_path = q.get()
			if i is None:
				break
			img = cv2.cvtColor((cv2.resize(cv2.imread(img_path, cv2.CV_LOAD_IMAGE_COLOR), (448, 448))), 
				cv2.COLOR_BGR2RGB)

			with lock:
				if i % 1000 == 0:
					print("processing %i/%i" % (i, len(images_path)))
				images[i] = img
			q.task_done()

	for _ in range(num_workers):
		thread = Thread(target=_worker)
		thread.daemon = True
		thread.start()
	q.join()

	print("Terminating threads...")
	for _ in range(2*num_workers):
		q.put((None, None))

	torch.save(images_idx, os.path.join(data_path, "%s_%s.pt" % (data_name, image_type)))
	dataset.close()
	print("Finish saving images...")


def main(opt):
	"""
	Create file that stores images in "train", "val", "trainval", and "test" datasets.
	"""
	# transform = transforms.Compose([
	# 		transforms.Scale(opt.size_scale),
	# 		transforms.ToTensor(),
	# 		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	# 	])

	# Process train images
	print("Create train images dataset...")
	save_images(opt.img_path, "train", opt.data_path, opt.data_name, opt.num_workers)

	# Process val images
	print("Create val images dataset...")
	save_images(opt.img_path, "val", opt.data_path, opt.data_name, opt.num_workers)

	# # Process trainval images
	# print("Create trainval images dataset...")
	# save_images(opt.img_path, "trainval", opt.data_path, opt.data_name, opt.num_workers)

	# # Process test images
	# print("Create test images dataset...")
	# save_images(opt.img_path, "test", opt.data_path, opt.data_name, opt.num_workers)

	print("Done!")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--img_path", default="/ceph/kien/data2.0")
	parser.add_argument("--data_name", default="cocoimages")
	parser.add_argument("--data_path", default="/ceph/kien/VQA/dataset")
	parser.add_argument("--num_workers", type=int, default=8)

	args = parser.parse_args()
	params = vars(args)
	print("Parsed input parameters:")
	print(json.dumps(params, indent=2))

	main(args)