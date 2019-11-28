# import Libraries
import argparse
import numbers
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as f
from torchvision.models import vgg16
from PIL import Image
import torch.optim as optim
from tqdm.autonotebook import tqdm
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset

# load custom functions
from train import *
from utils import *
from models import FCN32, FCN16


if __name__ == "__main__":
    # define default parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--data-path', type=str, default='./')
    parser.add_argument('--model', type=str, default='fcn32')
    parser.add_argument('--best-model', type=str,
                        default='baseline')
    parser.add_argument('--gpu', default=0)
    args = parser.parse_known_args()[0]

    # set random seed
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    np.random.seed(7)

    # set cuda device
    torch.cuda.empty_cache()
    torch.cuda.set_device(args)
    
    # create train,val dataloaders
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    target_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        RemoveContour(),
    ])
    
    trainset = torchvision.datasets.VOCSegmentation(
        './', image_set='train', transform=transform, target_transform=target_transform, download=False)
    validationset = torchvision.datasets.VOCSegmentation(
        './', image_set='val', transform=transform, target_transform=target_transform, download=False)
        
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=4,
                                              pin_memory=True)
    valloader = torch.utils.data.DataLoader(validationset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            pin_memory=True)

    # get some random training images
    (img, mask) = next(iter(trainloader))
    # unnormalize for visualization
    img = unnormalize(img)
    # visualize sample images and masks
    fig, ax = plt.subplots(2, 4, figsize=(8, 4))
    rows = ['original', 'groundtruth']
    for axs, row in zip(ax[:, 0], rows):
        axs.set_ylabel(row, rotation=0, labelpad=48, size='large')
        for i, (im, mk) in enumerate(zip(img[:4], mask[:4])):
            ax[0][i].imshow(np.moveaxis(im.numpy(), 0, -1))
            ax[0][i].set_yticklabels([])
            ax[0][i].set_xticklabels([])
            ax[1][i].imshow(mk)

    # import network
    if args.model == 'fcn16':
        net = FCN16(num_classes=21)
    else:
        net = FCN32(num_classes=21)
    trainable_params = sum(p.numel()
                           for p in net.parameters() if p.requires_grad)
    print("Total # trainable params: ", trainable_params)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net, [0, 1]).cuda()
    else:
        net = net.cuda()

    # define optimization functions
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), args.lr)
    torch.backends.cudnn.benchmark = True

    train(args, net, trainloader, valloader, criterion, optimizer)
