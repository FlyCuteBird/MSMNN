# Multi-Scale Motivated Neural Network for image-text matching (MPMNN) 


## Introduction
This is the source code of MSMNN. It is built on top of the SCAN (https://github.com/kuanghuei/SCAN) in PyTorch. In our experiments, we use two NVIDIA Tesla K80 GPUs for the parallel training. 
* python == 2.7
* [PyTorch](http://pytorch.org/) 0.3.0
* numpy >= 1.12.1

## Download data
Download the dataset files. We use the visual local features created by SCAN, downloaded [here](https://github.com/kuanghuei/SCAN), and salience features can be extracted by R^3 net (https://github.com/zijundeng/R3Net). After extracting the salience features, we need to connect local features and salience features by the order of images.

## Training

```bash
python train.py
```
## Evaluation
```bash
python evalution1.py
```



