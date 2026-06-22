import collections
import copy
from tqdm import tqdm
import torch
from torch import nn
import numpy as np 
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

def fed_prox_update_from_local(global_model, client_loader, test_loader, num_local_epochs, optimization_args):
    lambda_= 0.01
    local_model = copy.deepcopy(global_model)
    ### Starting the training process
    local_model.train()
    device = local_model.device ## Using the same device local model is uisng to store 
    optimizer = torch.optim.Adam(local_model.parameters(), **optimization_args) ## Using the Adam Optimization that takes the kwargs of the optimizer_args dictionary which can contain any parameters
    loss_function = nn.CrossEntropyLoss() ## Using the cross entropy loss

    for epoch in range(num_local_epochs): ## Looping through each epch 
        ## Using the tqdm to track the progess bar virtually 
        for (x,y) in tqdm(client_loader, desc = 'epoch {}/{}'.format(epoch+1, num_local_epochs)): ## In number of batches 
            optimizer.zero_grad() ## Clearing out the gradient  
            x = x.to(device) 
            y = y.to(device)
            loss = loss_function(local_model(x), y) ## Calculating the loss
            reg_loss = 0.0
            for param,global_param in zip(local_model.parameters(), global_model.parameters()):
                reg_loss += torch.norm(param-global_param, p = 2) ** 2
            total_loss = loss + lambda_ / 2 * reg_loss
            total_loss.backward() ## Back propagation 
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
### This is used when the model asks for the weight update: FedNova


def fednova_weight_averaging(global_model, weight_list, num_samples, tau_k,device, learning_rate):
    global_weights = global_model.state_dict()
    total_tau = sum(tau_k) ## Total number of samples
    keys = weight_list[0].keys()
    total_samples = sum(num_samples)

    p_i = [num/ total_samples for num in num_samples]

    tau_efficient = sum(p_i[i] * tau_k[i] for i in range(len(weight_list)))
    normalized_weight_average = collections.OrderedDict()

    for k in keys:
        normalized_weight_average[k] = torch.zeros(weight_list[0][k].size()).to(device)
    for i in range(len(weight_list)): ## This is the number of clients
        client_contribution = p_i[i]/tau_k[i]
        for k in keys:
            normalized_weight_average[k] += client_contribution * (weight_list[i][k]).to(device)
    new_global_weights = collections.OrderedDict()
    for k in keys:
        new_global_weights[k] = global_weights[k] - learning_rate * tau_efficient* normalized_weight_average[k]
    return new_global_weights


def fednova_update_from_local(global_model, client_loader, test_loader, num_local_epochs, optimizer_args):
    local_model = copy.deepcopy(global_model) 
    local_model.train()
    device = local_model.device
    optimizer = torch.optim.SGD(local_model.parameters(), lr=optimizer_args['lr'], momentum=0.9)
    loss_function = nn.CrossEntropyLoss() ## Using the cross entropy loss
    tow_k = 0 ## Here tow_k stores the number of gradients performed by each client to adjust the weight/ number of total batches 
    for epoch in range(num_local_epochs):
        epoch_loss = 0
        for (x,y) in tqdm(client_loader, desc = 'epoch {}/{}'.format(epoch+1, num_local_epochs)):
        
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            loss = loss_function(local_model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            tow_k+= 1
        print(f"Epoch {epoch+1} avg loss: {epoch_loss/len(client_loader):.4f}") 
        
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



### Purposed Solution: Hierarchical Federated Nova Weight Calculation
def hierar_fednova_weight_averageing(global_model, weight_list, num_samples, tau_k, device, alpha = 0.8,  server_learning_rate = 2.0):
    global_weights = global_model.state_dict()
    keys = weight_list[0].keys()
    total_samples = sum(num_samples)

    ### Defining the groups based on their total data contribution 
    large_clients = [0,1]
    small_clients = [2,3]

    large_samples = sum(num_samples[i] for i in large_clients) ## Total samples of data in the large clients
    large_clients_weight_averaging = collections.OrderedDict()

    for k in keys:
        large_clients_weight_averaging[k] = torch.zeros(weight_list[0][k].size()).to(device)
    for i in large_clients:
        large_each_contribute = (num_samples[i]/large_samples) / tau_k[i]
        for k in keys:
            large_clients_weight_averaging[k] += large_each_contribute * weight_list[i][k].to(device)
    ### Doing same for small clients

    small_samples = sum(num_samples[i] for i in small_clients)
    small_clients_weight_averaging = collections.OrderedDict()

    for k in keys:
        small_clients_weight_averaging[k] = torch.zeros(weight_list[0][k].size()).to(device)
    for i in small_clients:
        small_each_contribute = (num_samples[i]/small_samples) / tau_k[i]
        for k in keys:
            small_clients_weight_averaging[k] += small_each_contribute * weight_list[i][k].to(device)
    
    ### Combining both of them : 

    new_global_weights = collections.OrderedDict()
    for k in keys:
        new_global_weights[k] = global_weights[k] - server_learning_rate * (alpha * large_clients_weight_averaging[k] + (1-alpha) * small_clients_weight_averaging[k])
    
    return new_global_weights