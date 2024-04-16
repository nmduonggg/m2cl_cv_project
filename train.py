import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from collections import OrderedDict
import numpy as np
from loss import my_loss
from model import M2CL18
from data.DataLoader import get_train_dataloader, augment_transform, get_test_loader
import argparse
from test import do_test

lr = 0.01
num_epochs = 10
availabel_dataset = ["dslr", "amazon", "webcam"]
def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", "-l", type=float, default= 0.01, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=31, help="Number of classes")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation size (between 0 and 1)")
    parser.add_argument("--source", choices=availabel_dataset, help="Training dataset")
    parser.add_argument("--target", choices=availabel_dataset, help="Test dataset" )
    return parser.parse_args()
if __name__ == "__main__":
    args = get_args()

    network = M2CL18(args.n_classes, pretrained=True)
    optimizer = torch.optim.SGD(
            network.parameters(),
            lr=args.learning_rate,
            weight_decay=0.0005,
            momentum=0.9
        )
    trainloader, valloader = get_train_dataloader(args.source,args.batch_size,args.val_size, augment_transform)
    testloader = get_test_loader(args.target, args.batch_size)
    # x = torch.rand((10,3,224,224))
    # y = torch.Tensor([0,0,1,1,1,2,3,4,2,3]).long()
    # preds, conv_act = network(x)
    
    for epoch in range(args.epochs):
        network.train()
        train_loss = 0
        true_pred = 0
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
        print(f"Training loss: {train_loss/len(trainloader.dataset)}, accuracy: {true_pred/len(trainloader.dataset)}")

        ##Validation
        network.eval()
        val_loss_epoch = 0
        for x,y in valloader:
            preds, conv_act = network(x)
            val_loss = F.cross_entropy(preds, y)
            val_loss_epoch += val_loss
        val_loss_epoch = val_loss_epoch / len(valloader.dataset)
        print(f"Validation loss: {val_loss_epoch}")

    test_loss = do_test(network, testloader)
    