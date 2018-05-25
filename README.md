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
This repository contains Pytorch implementation of "[Improved Fusion of Visual and Language Representations by Dense Symmetric Co-Attention for Visual Question Answering](https://arxiv.org/abs/1804.00775)" paper.

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

## Running
The dataset can be downloaded from: [http://visualqa.org/](http://visualqa.org/).
All of arguments are described in the train and answer file.
Run train.py file for the training and answer.py for generating answers.

## License
The source code is licensed under [GNU General Public License v3.0](./LICENSE).
