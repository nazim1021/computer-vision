# import Libraries
import argparse
import numbers
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as f
from torchvision.models import vgg16
from PIL import Image
import torch.optim as optim
from tqdm.autonotebook import tqdm
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset

# load custom functions
from utils import *
from models import FCN32, FCN16

if __name__ == "__main__":

    # define default parameters
    parser = argparse.ArgumentParser()
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

    # load the best model for evaluation
    if args.model =='fcn16':
        net = FCN16(num_classes=21)
    else:
        net = FCN32(num_classes=21)

    for param in net.parameters():
        param.requires_grad = False

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net, [0, 1]).cuda()
    else:
        net = net.cuda()

    model = torch.load(args.best_model+'.pth')
    net.load_state_dict(model['model_state_dict'])

    testset = torchvision.datasets.VOCSegmentation(
        './', image_set='val', transform=transform, target_transform=target_transform, download=False)

    # load validation dataset to be used for testing
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=4,
                                             pin_memory=True)

    net.eval()
    preds = []
    targets = []
    imgs = []
    with torch.no_grad():
        for (images, labels) in testloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            predicted = torch.argmax(outputs, 1)
            preds.extend(list(predicted.cpu().numpy()))
            targets.extend(list(labels.cpu().numpy()))

    # calculate overall iou and dice score
    preds = np.array(preds)
    targets = np.array(targets)
    cm = confusion_matrix(preds.flatten(), targets.flatten())
    evaluate(cm)

    # unnormalize for visualization
    imgs = unnormalize(images.cpu())

    # visualize sample predictions
    fig, ax = plt.subplots(4, 4, figsize=(10, 8))
    rows = ['original', 'groundtruth', "prediction"]

    for axs, row in zip(ax[:, 0], rows):
        axs.set_ylabel(row, rotation=0, labelpad=48, size='large')
    for i, (im, pd, gt) in enumerate(zip(imgs[-
                                              4:], predicted[:4].cpu().numpy(), labels[:4].cpu().numpy())):
        ax[0][i].imshow(np.moveaxis(im.numpy(), 0, -1))
        ax[0][i].set_yticklabels([])
        ax[0][i].set_xticklabels([])
        ax[1][i].imshow(get_label_colormap(gt))
        ax[1][i].set_yticklabels([])
        ax[1][i].set_xticklabels([])
        ax[2][i].imshow(get_label_colormap(pd))
        ax[2][i].set_yticklabels([])
        ax[2][i].set_xticklabels([])
