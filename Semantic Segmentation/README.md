# Semantic Segmentation using Fully Convolutional Networks
Pytorch implementation of the paper [Fully Convolutional Networks for Semantic Segmentation](http://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf). 

## Requirements
1. Python 3.6 + 
2. Python 1.0 +
3. Numpy
4. Matplotlib

## Dataset
PASCAL VOC (2012) dataset consisting of 20 different objects:

['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog',
'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv']

We load the dataset using pytorch’s inbuilt `torchvision.datasets.VOCSegmentation()` method

## Usage
1. Training: 
```
python main.py --num-epochs=15 --lr=0.01 --batch-size=16 --model=fcn16
```
- Model can be either `fcn16` or `fcn32`. The file also contains code for visualization of sample images. 

2. Evaluation:
```
python test.py --batch-size=16 --model=fcn16 --best-model=baseline
```

## Results

<img src=results.png>


## References
1. Long, J, et.al "Fully Convolutional Networks for Semantic Segmentation", CVPR 2015.
2. http://deeplearning.net/tutorial/fcn_2D_segm.html