# Heterogeneous Graph Fusion Network for Cross-modal Image-text Retrieval 


## Introduction
This is the source code of Heterogeneous Graph Fusion Network . It is built on top of the SCAN (https://github.com/kuanghuei/SCAN) in PyTorch. In our experiments, we use two NVIDIA Tesla K80 GPUs for the parallel training 
* [PyTorch](http://pytorch.org/) 0.3.0
* numpy >= 1.12.1

## Download data
Download the dataset files. We use the image feature created by SCAN, downloaded [here](https://github.com/kuanghuei/SCAN), and salience features can be extracted by  R^3 net

## Training

```bash
python train.py
```
## Evaluation
```bash
python evalution1.py
```



