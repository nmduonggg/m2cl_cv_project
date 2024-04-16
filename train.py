import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from collections import OrderedDict
import numpy as np
from loss import my_loss
from model import M2CL18
from data.DataLoader import get_train_dataloader, augment_transform


lr = 0.01
num_epochs = 10

if __name__ == "__main__":
    
    network = M2CL18(31, pretrained=True)
    optimizer = torch.optim.SGD(
            network.parameters(),
            lr=lr,
            weight_decay=0.0005,
            momentum=0.9
        )
    trainloader = get_train_dataloader('dslr', augment_transform, 4)

    # x = torch.rand((10,3,224,224))
    # y = torch.Tensor([0,0,1,1,1,2,3,4,2,3]).long()
    # preds, conv_act = network(x)
    
    for epoch in range(num_epochs):
        train_loss = 0
        true_pred = 0
        samples_count = 0
        for x,y in trainloader:
            
            preds, conv_act = network(x)
            y_tmp_np = y.cpu().detach().numpy()
            y_tmp = y_tmp_np.tolist()
            counts = {}
            same_indexes_tmp = {}
            dif_indexes = {}
            for i in y_tmp:
                counts[i] = y_tmp.count(i)
                same_indexes_tmp[i] = np.where(y_tmp_np == i)
                dif_indexes[i] = np.where(y_tmp_np != i)

            same_indexes_tmp = OrderedDict(sorted(same_indexes_tmp.items()))
            same_indexes = []
            for i in range(len(same_indexes_tmp.items())):
                if i in same_indexes_tmp.keys():
                    same_indexes.append(torch.combinations(torch.tensor(same_indexes_tmp[i][0])))

            custom_loss = my_loss(conv_act,
                                same_indexes,
                                0.01,
                                1.0)

            ce_loss = F.cross_entropy(preds, y)

            loss = custom_loss + ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            y_pred = torch.argmax(preds, 1)
            # print(f"y pred is {y_pred} and y is {y}")
            true_pred += torch.sum(y_pred == y).item()
            samples_count += y.shape[0]
            print(y.shape[0])
        print(f"Training loss: {train_loss}, accuracy: {true_pred/len(trainloader)}")
        print(ce_loss)
            