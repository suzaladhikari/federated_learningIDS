### Here we will be doing all the steps required to train on nids dataset 

import torch
from torchvision.transforms import functional as TF
from torchvision import transforms
from torch.utils.data import Dataset

import os
import sys

import numpy as np
import collections
from tqdm import tqdm 
import utils

def evaluate_model(model, data_loader, loss_function, tqdm_desc = None, seed = 42):
    device = model.device
    loss_metric = utils.MeanMetric()
    with eval_model(model)