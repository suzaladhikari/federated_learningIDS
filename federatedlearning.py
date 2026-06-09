import collections
import copy
from tqdm import tqdm
import torch
from torch import nn

def updatefrom_local(global_model, client_loader, test_loader, num_local_epohcs, optimizier_args):
    local_model = copy.deepcopy(global_model) ## Copying the global model and use it in the local clients
    ### Starting the training process
    local_model.train()
    device = local_model.device ## Using the same device local model is uisng to store 
    optimizer = torch.optim.Adam(local_model.parameters(), **optimizier_args) ## Using the Adam Optimization that takes the kwargs of the optimizer_args dictionary which can contain any parameters
    loss_function = nn.CrossEntropyLoss() ## Using the cross entropy loss


