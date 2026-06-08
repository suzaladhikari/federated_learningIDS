import os 
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


### Feature One: Creating object for saving the given objects in pickle format:

def savein_pickle(path, object):
    with open(path, 'wb') as f:
        pickle.dump(object, f)
    

### Feature Two: Utlis for loading the objecst from the pickle format 
def loading_pickle(filepath):
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj 
