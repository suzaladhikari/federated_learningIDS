### Here we will be doing all the steps required to train on nids dataset 

import torch
from torchvision.transforms import functional as TF
from torchvision import transforms
from torch.utils.data import Dataset
from contextlib import contextmanager
import os
import sys

import numpy as np
import collections
from tqdm import tqdm 
import utils
@contextmanager
def train_model(model,mode = True):
    modes = [module.training for module in model.modules()] ## This stores True if the layer is in training mode and false if the layer is in evaluation mode.
    try: ## Tell model to try doing this
        yield model.train(mode) ## If mode if False then it goes to eval if true goes to training
    finally: ## Doesnot matter the condition model has to do this 
        for i, module in enumerate(model.modules()):
            module.training = modes[i]

def eval_mode(model):
    return train_model(model, False)
def evaluate_model(model, data_loader, loss_function, tqdm_desc = None, seed = 42):
    device = model.device
    loss_metric = utils.MeanMetric()
    with eval_model(model):
