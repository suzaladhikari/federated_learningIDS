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
    modes = [module.training for module in model.modules()] ## Saving the original state of the model, True or False means active or inactive
    try: ## Tell model to try doing this
        yield model.train(mode) ## If mode if False then it goes to eval if true goes to training
    except Exception as e: 
        print(f"Something went wrong during evaluation{e}") ## Playing the safe game of returning what went wrong during evaluation 
        raise
    finally: ## Doesnot matter the condition model has to do this 
        for i, module in enumerate(model.modules()):
            module.training = modes[i] ## Storing back the original state 

def eval_mode(model): ## Putting the training model into evaluation temporarily
    return train_model(model, False)


@torch.no_grad()
def evaluate_model(model, data_loader, loss_function, tqdm_desc = None, seed = 42):
    device = model.device ## setting up the device 
    loss_metric = utils.MeanMetric() ## Setting upthe loss 
    with eval_model(model): ## This calls the eval mode
        torch.manual_seed(seed)
        for (x,y) in tqdm(data_loader,desc = tqdm_desc ):
            x = x.to(device)
            y = y.to(device)
            loss = loss_function(model(x), y)
            loss_metric.update_state(loss.item())

        return loss_metric.result()
    

## Everything the evaulate_model is doing is temporarily evaulating them model and if something goes wrong boom restore the model's original state 

    


