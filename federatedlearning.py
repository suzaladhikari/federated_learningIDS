import collections
import copy
from tqdm import tqdm
import torch
from torch import nn
from nids_training import evaluate_model

### Averaging the weights 

def weight_averaging(weight_list, num_sample_list, device):
    total_samples = sum(num_sample_list) ## Sum of all the weights of all the layers of all the clients 
    keys = weight_list[0].keys() ## Since all of the clients share the same layer, we are extracting the keys
    weight_average = collections.OrderedDict() ## Creating a dictionary to store updated weights

    for k in keys:
        weight_average[k] = torch.zeros(weight_list[0][k].size()).to(device) ## Creating the weight average dictionary which stores all the zeros of the size of the layers and bias 
    for k in keys:
        for i in range(len(weight_list)): ## Go to each client
            weight_average[k] += weight_list[i][k] * num_sample_list[i] ## Average their weight times the samples
        weight_average[k] = torch.div(weight_average[k], total_samples) ## Dividing the whole mulitiplication by the total samples count 
        weight_average[k]
    return weight_average

def fednova_weight_averaging(weight_list, num_samples, tau_k,device):
    total_samples = sum(num_samples) ## Total number of samples
    keys = weight_list[0].keys()
    weight_average = collections.OrderedDict()
    for k in keys:
        weight_average[k] = torch.zeros(weight_list[0][k].size()).to(device)
    

def updatefrom_local(global_model, client_loader, test_loader, num_local_epohcs, optimizier_args):
    local_model = copy.deepcopy(global_model) ## Copying the global model and use it in the local clients
    ### Starting the training process
    local_model.train()
    device = local_model.device ## Using the same device local model is uisng to store 
    optimizer = torch.optim.Adam(local_model.parameters(), **optimizier_args) ## Using the Adam Optimization that takes the kwargs of the optimizer_args dictionary which can contain any parameters
    loss_function = nn.CrossEntropyLoss() ## Using the cross entropy loss

    for epoch in range(num_local_epohcs): ## Looping through each epch 
        ## Using the tqdm to track the progess bar virtually 
        for (x,y) in tqdm(client_loader, desc = 'epoch {}/{}'.format(epoch+1, num_local_epohcs)): ## In number of batches 
            optimizer.zero_grad() ## Clearing out the gradient  
            x = x.to(device) 
            y = y.to(device)
            loss = loss_function(local_model(x), y) ## Calculating the loss
            loss.backward() ## Back propagation 
            optimizer.step() ## Optimization of the model weights
        
    training_loss = evaluate_model(local_model, client_loader, loss_function, tqdm_desc='Local Training Loss')   ## The updated weight model is sent for further evaluation. 
    testing_loss = evaluate_model(local_model, test_loader, loss_function, tqdm_desc='Local Testing Loss')
            
      ## Returning the dictionary 
    local_update = {
        'local_weights': local_model.state_dict(),
        'num_samples': len(client_loader.dataset),
        'train_loss': training_loss,
        'test_loss': testing_loss  
      }      
    
    return local_update


def fednova_update_from_local(global_model, client_loader, test_loader, num_local_epochs, optimizer_args):
    local_model = copy.deepcopy(global_model) 
    local_model.train()
    device = local_model.device
    optimizer = torch.optim.Adam(local_model.parameters(), **optimizer_args)
    loss_function = nn.CrossEntropyLoss() ## Using the cross entropy loss
    tow_k = 0 ## Here tow_k stores the number of gradients performed by each client to adjust the weight/ number of total batches 
    for epoch in range(num_local_epochs):
        for (x,y) in tqdm(client_loader, desc = 'epoch {}/{}'.format(epoch+1, num_local_epochs)):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            loss = loss_function(local_model(x), y)
            tow_k+= 1
            loss.backward()
            optimizer.step()
        
    delta_weights = {} ## Storing how much the local weight has been changed by the client
    global_weights = global_model.state_dict()
    local_weights = local_model.state_dict()
    for key in global_weights.keys(): ### Caculating the change in weight in each layer
        delta_weights[key] = global_weights[key]- local_weights[key]

    training_loss = evaluate_model(local_model, client_loader, loss_function, tqdm_desc='Local Training Loss')   ## The updated weight model is sent for further evaluation. 
    testing_loss = evaluate_model(local_model, test_loader, loss_function, tqdm_desc='Local Testing Loss')

    local_update = {
        'tau_k': tow_k,
        'delta_weights': delta_weights,
        'num_samples': len(client_loader.dataset),
        'training_loss': training_loss,
        'testing_loss': testing_loss
    }

    return local_update

