# Improved Fusion of Visual and Language Representations by Dense Symmetric Co-Attention for Visual Question Answering

If you make use of this code, please cite the following paper (and give us a star ^_^):
```
@InProceedings{Nguyen_2018_CVPR,
author = {Nguyen, Duy-Kien and Okatani, Takayuki},
title = {Improved Fusion of Visual and Language Representations by Dense Symmetric Co-Attention for Visual Question Answering},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```

If you have any suggestion to improve this code, please feel free to contact me at ```kien@vision.is.tohoku.ac.jp```.

## Overview
This repository contains Pytorch implementation of "[Improved Fusion of Visual and Language Representations by Dense Symmetric Co-Attention for Visual Question Answering](https://arxiv.org/abs/1804.00775)" paper. The network architecture is illustrated in Figure 1.

![Figure 1: Overview of Dense Co-Attention Network architecture.](imgs/dcn.png)
<center>Figure 1: The Dense Co-Attention Network architecture.</center>

## Files
```
├──vqa_eval/ - Evaluation code provided from VQA team
├──preprocess/ - Preprocessing code before training the network
├──dense_coattn/ - Dense Co-Attention code
├──demo/ - Demo code for pretrained Dense Co-Attention model
train.py - Train the model
answer.py - Generate the answer for test dataset
ensemble.py - Ensemble multiple results from different models
vqa_eval.py - Evaluate the performance of model (Provided by VQA team)
```

## Dependencies
Tests are performed with following version of libraries:

+ Python 3.6.3
+ Numpy 1.13.3
+ Pytorch 0.3.1
+ Torchtext 0.2.1
+ TensorboardX

## Demo
1. Download our pretrained model and data info: [download link](https://drive.google.com/drive/folders/1Qvxu2ZMfPBkVL3gqdBV0oupY22F3sKLU) and put it inside demo folder.
2. Run ```demo/single_machine_demo.py``` to test our pretrained mode (if you want to run on cpu, specify --gpus is an empty list).

## Training from Scratch
The dataset can be downloaded from: [http://visualqa.org/](http://visualqa.org/).

We provide the scripts for training our network from scratch by simply running the ```train.py``` script to train the model. 

- All of arguments are described in the ```train.py``` file so that you can easily change the hyper-parameter and training conditions (Most of the default hyper-parameters are used in the main paper).
- Pretrained GloVe word embedding is loaded from [torchtext](https://github.com/pytorch/text) (torchtext for Pytorch 0.3)

## Evaluation

Run ```answer.py``` file to generate all of answers for the test set. You can use ```ensemble.py``` to ensemble multiple model's results for the evaluation

## License
The source code is licensed under [MIT](./LICENSE).
