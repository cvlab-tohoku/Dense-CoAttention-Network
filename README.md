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
├──preprocess/ - Preprocessing code before training the network
├──dense_coattn/ - Dense Co-Attention code
├──demo/ - Demo imgs for pretrained Dense Co-Attention model
train.py - Train the model
answer.py - Generate the answer for test dataset
ensemble.py - Ensemble multiple results from different models
```

## Dependencies
Tests are performed with following version of libraries:

+ Python 3.6.3
+ Pytorch >= 0.4
+ Torchtext for Pytorch >= 0.4 (install via pip)
+ TensorboardX

## Training from Scratch
The dataset can be downloaded from: [http://visualqa.org/](http://visualqa.org/).

We provide the scripts for training our network from scratch by simply running the ```train.py``` script to train the model. 

- All of arguments are described in the ```train.py``` file so that you can easily change the hyper-parameter and training conditions (Most of the default hyper-parameters are used in the main paper).
- Pretrained GloVe word embedding is loaded from [torchtext](https://github.com/pytorch/text)

## Evaluation

Run ```answer.py``` file to generate all of answers for the test set. You can use ```ensemble.py``` to ensemble multiple model's results for the evaluation

## License
The source code is licensed under [MIT License](./LICENSE).
