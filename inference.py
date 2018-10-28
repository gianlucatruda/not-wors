import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import sys
from tools import vis_holdout, vis_demo

if __name__ == '__main__':
    args = sys.argv
    if len(args) != 3:
        print("Incorrect parameters")
        sys.exit()

    fpath = args[1]
    mode = args[2].lower()

    if mode != 'demo':
        print("Running in evaluation mode on holdout data")
    else:
        print("Running in demo mode on unseen data")

    print("Loading model from: "+fpath)

    model_conv = torchvision.models.resnet18(pretrained=True)

    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    # Restore a saved model
    state_dict = torch.load(fpath)
    model_conv.load_state_dict(state_dict)

    print("Loaded model.")
    print("Evaluating on holdout...")

    if mode == "demo":
        vis_demo(model_conv)
    else:
        vis_holdout(model_conv)
    plt.pause(0.001)
    plt.ioff()
    plt.show()


