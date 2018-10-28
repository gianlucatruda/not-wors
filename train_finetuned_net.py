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
    Fine-tuned CNN

    Instead of random initializaion, we initialize the network with a
    pretrained network, like the one that is trained on imagenet 1000
    dataset. Rest of the training looks as usual.
    """

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    print("Fine-tuning model...")
    model_ft = train_model(
        model_ft, criterion, optimizer_ft,
        exp_lr_scheduler,
        num_epochs=1
    )

    visualize_model(model_ft)

    print("Saving model for later inference...")
    now = datetime.datetime.now().strftime('%I%M')
    torch.save(model_ft.state_dict(), 'models/ft_'+now+'.model')

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
