import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loss import my_loss
from model import M2CL18
from tqdm import tqdm

def do_test(network, dataloader, device = 'cpu'):
    network.to(device)
    network.eval()
    test_loss = 0
    true_pred = 0
    with tqdm(total=len(dataloader)) as pbar:
        for x, y in dataloader:
            x,y = x.to(device), y.to(device)
            preds, conv_act = network(x)
            loss = F.cross_entropy(preds, y)
            test_loss += loss.detach()
            test_loss = test_loss / len(dataloader.dataset)
            y_pred = torch.argmax(preds, 1)
            true_pred += torch.sum(y_pred == y).item()

            del loss, preds, conv_act
            
            pbar.update(1)
    
    print(f"Test loss on new domain: {test_loss}")
    print(f"Accuracy on new domain is: {true_pred / len(dataloader.dataset)}")
        
    return test_loss
def do_test_resnet(network, dataloader,device='cpu'):
    network.to(device)
    network.eval()
    test_loss = 0
    true_pred = 0
    with tqdm(total=len(dataloader)) as pbar:
        for x, y in dataloader:
            x,y = x.to(device), y.to(device)
            preds = network(x)
            loss = F.cross_entropy(preds, y)
            test_loss += loss.detach()
            test_loss = test_loss / len(dataloader.dataset)
            y_pred = torch.argmax(preds, 1)
            true_pred += torch.sum(y_pred == y).item()

            del loss, preds
            
            pbar.update(1)
    
    print(f"Test loss on new domain: {test_loss}")
    print(f"Accuracy on new domain is: {true_pred / len(dataloader.dataset)}")
        
    return test_loss
