import torch 
from torch import nn
import math 

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc_layers = 
