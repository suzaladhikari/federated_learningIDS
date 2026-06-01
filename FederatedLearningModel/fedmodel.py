import torch 
from torch import nn
import math 

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc_layers = nn.ModuleList() ## We are using nn.ModuleList instead of nn.Sequential becuase we need dynamic control over the layer construction and forward execution.
        layer_sizes = [input_size] + hidden_size 
        for i in range(len(hidden_size)):
            self.fc_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))


