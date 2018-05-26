# Improved Fusion of Visual and Language Representations by Dense Symmetric Co-Attention for Visual Question Answering

If you make use of this code, please cite the following paper:
```
@article{DBLP:journals/corr/abs-1804-00775,
  author    = {Duy{-}Kien Nguyen and Takayuki Okatani},
  title     = {Improved Fusion of Visual and Language Representations by Dense Symmetric Co-Attention for Visual Question Answering},
  journal   = {CoRR},
  volume    = {abs/1804.00775},
  year      = {2018},
  url       = {http://arxiv.org/abs/1804.00775}
}
```
## Overview
This repository contains Pytorch implementation of "[Improved Fusion of Visual and Language Representations by Dense Symmetric Co-Attention for Visual Question Answering](https://arxiv.org/abs/1804.00775)" paper. The network architecture is illustrated in Figure 1.

![Figure 1: Overview of Dense Co-Attention Network architecture.](imgs/dcn.png)
<center>Figure 1: The Dense Co-Attention Network architecture.</center>

## Files
```
├──vqa_eval/ - Evaluation code provided from VQA team
├──preprocess/ - Preprocessing code before training the network
├──dense_coattn/ - Dense Co-Attention code
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
+ Torchtext (for Pytorch 0.3)
+ Tensorflow (if you want to use tensorboard visualization)

## Training from Scratch
The dataset can be downloaded from: [http://visualqa.org/](http://visualqa.org/).

We provide the scripts for training our network from scratch by simply running the ```train.py``` script to train the model. 

- All of arguments are described in the ```train.py``` file so that you can easily change the hyper-parameter and training conditions (Most of the default hyper-parameters are used in the main paper).
- Pretrained GloVe word embedding is loaded from [torchtext](https://github.com/pytorch/text) (torchtext for Pytorch 0.3)

## Evaluation

Run ```answer.py``` file to generate all of answers for the test set. You can use ```ensemble.py``` to ensemble multiple model's results for the evaluation

## License
The source code is licensed under [GNU General Public License v3.0](./LICENSE).
