import torch 
from torch import nn
import math 

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc_layers = nn.ModuleList() ## We are using nn.ModuleList instead of nn.Sequential becuase we need dynamic control over the layer construction and forward execution.
        layer_sizes = [input_size] + hidden_size 
        for i in range(len(hidden_size)):
            self.fc_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1])) ## Lets assume the input is 1568 and hidden layer is [512,64,32,9]
            ### In such way the layers become (1568,512), (512,64), (64,32), (32,9)
            ### Between each layers we adapt the activation function oif leakyRely with negative slope of 0.001
            self.fc_layers.append(nn.LeakyReLU(negative_slope=0.001))
        self.fc_layers.append(nn.Linear(hidden_size[-1], num_classes)) ## This is used as the dense layer where we just convert the given last hidden layer to the number of classes 
    
    def forward(self,x):
        for layer in self.fc_layers:
            x = layer(x) ### Here x stores the output for each layer 
        return x 
    

    @property 
    def device(self):
        return next(self.parameters()).device


