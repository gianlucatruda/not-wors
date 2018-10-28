"""
This is the code to train a ResNet to identify boerewors using
transfer learning.

This code is adapted from the following tutorial:

TRANSFER LEARNING TUTORIAL
Author: Sasank Chilamkurthy

https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py

"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import datetime
import os
import copy

from tools import imshow as imshow
from tools import train_model, visualize_model

import warnings

def main():
    # warnings.filterwarnings('ignore')

    plt.ion()   # interactive mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """
    ------------------------------------------------------------------------
    ConvNet as fixed feature extractor

    Here, we will freeze the weights for all of the network except that
    of the final fully connected layer. This last fully connected layer
    is replaced with a new one with random weights and only this layer
    is trained.
    """

    print("Training CNN as fixed feature extractor...")

    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


    model_conv = train_model(model_conv, criterion, optimizer_conv,
                            exp_lr_scheduler, num_epochs=1)


    visualize_model(model_conv)

    print("Saving model for later inference...")
    now = datetime.datetime.now().strftime('%I%M')
    torch.save(model_conv.state_dict(), 'models/ffe_'+now+'.model')


    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
