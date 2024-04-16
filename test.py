import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loss import my_loss
from model import M2CL18

def do_test(network, dataloader):
    network.eval()
    test_loss = 0
    for x,y in dataloader:
        preds, conv_act = network(x)
        loss = F.cross_entropy(preds, y)
        test_loss += loss
        test_loss = test_loss / len(dataloader.dataset)
    print(f"Test loss on new domain: {test_loss}")
    return test_loss