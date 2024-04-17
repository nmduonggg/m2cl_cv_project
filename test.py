import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loss import my_loss
from model import M2CL18

def do_test(network, dataloader):
    network.eval()
    test_loss = 0
    true_pred = 0
    for x,y in dataloader:
        preds, conv_act = network(x)
        loss = F.cross_entropy(preds, y)
        test_loss += loss
        test_loss = test_loss / len(dataloader.dataset)
        y_pred = torch.argmax(preds, 1)
        true_pred += torch.sum(y_pred == y).item()
    print(f"Test loss on new domain: {test_loss}")
    print(f"Accuracy on new domain is: {true_pred/len(dataloader.dataset)}")
    return test_loss