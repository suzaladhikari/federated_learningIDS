import collections
import copy
from tqdm import tqdm
import torch
from torch import nn
from nids_training import evaluate_model

### Averaging the weights 
def weight_averaging(weight_list, num_sample_list):
    total_samples = sum(num_sample_list)
    keys = weight_list[0].keys()
    weight_average = collections.OrderedDict()
    for k in keys:
        weight_average[k] = torch.zeros(weight_list[0][k].size()) ## Creating the weight average dictionary which stores all the zeros of the size of the layers and bias 
    for k in keys:
        for i in range(len(weight_list)):
            weight_average[k] += weight_list[i][k] * total_samples[i]
        weight_average[k]
    

def updatefrom_local(global_model, client_loader, test_loader, num_local_epohcs, optimizier_args):
    local_model = copy.deepcopy(global_model) ## Copying the global model and use it in the local clients
    ### Starting the training process
    local_model.train()
    device = local_model.device ## Using the same device local model is uisng to store 
    optimizer = torch.optim.Adam(local_model.parameters(), **optimizier_args) ## Using the Adam Optimization that takes the kwargs of the optimizer_args dictionary which can contain any parameters
    loss_function = nn.CrossEntropyLoss() ## Using the cross entropy loss

    for epoch in range(num_local_epohcs):
        ## Using the tqdm to track the progess bar virtually 
        for (x,y) in tqdm(client_loader, desc = 'epoch {}/{}'.format(epoch+1, num_local_epohcs)):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            loss = loss_function(local_model(x), y)
            loss.backward()
            optimizer.step()
        
    training_loss = evaluate_model(local_model, client_loader, loss_function, tqdm_desc='Local Training Loss')
    testing_loss = evaluate_model(local_model, test_loader, loss_function, tqdm_desc='Local Testing Loss')
            
      ## Returning the dictionary 
    local_update = {
        'local_weights': local_model.state_dict(),
        'num_samples': len(client_loader.dataset),
        'train_loss': training_loss,
        'test_loss': testing_loss  
      }      
    
    return local_update



