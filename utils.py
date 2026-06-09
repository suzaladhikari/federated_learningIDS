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


### Computing the accuracy of the given batch 

def accuracy_batch(y_batch, y_pred):
    _, predicted = torch.max(y_pred, 1)
    accuracy = (predicted == y_batch).sum().item() / len(y_batch)
    return accuracy

### For calculating the Mean Metric 

class MeanMetric():
    def __init__(self):
        self.total = np.float32(0)
        self.count = np.float32(0)

    def update_state(self, value):
        self.total += value
        self.count += 1 
    
    def result(self):
        if self.count > 0: 
            return self.total /self.count 
        else:
            return np.nan
    
    def reset_state(self):
        self.total = np.float32(0)
        self.count = np.float32(0)

def performance_analyzer(metric = {'train_loss', 'train_acc', 'valid_loss', 'valid_acc'}):
    performance_dict, performance_log = dict(), dict()
    for key in metric:
        performance_dict[key] = MeanMetric()
        performance_log[key] = list()
    return performance_dict, performance_log


### Creating a custom dataset that converts to the format needed while doing the data processing !

class CustomDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.features = self.data.iloc[:,:-1].values ## Every column except the last one
        self.labels = self.data.iloc[:,-1].values ## The last column 

        ## Label Encoding- If there are any string classes

        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(self.labels)

        ## Converting them to the tensor format 

        self.features = torch.FloatTensor(self.features)
        self.labels = torch.LongTensor(self.labels)

    
    