import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from collections import OrderedDict
import numpy as np
from loss import my_loss
from model import M2CL18, resnet18
from data.DataLoader import get_train_dataloader, augment_transform, get_test_loader
import argparse
from test import do_test, do_test_resnet
from tqdm import tqdm
availabel_dataset = ["dslr", "amazon", "webcam", "CALTECH", "LABELME", "PASCAL", "SUN", "art_painting", "cartoon", "photo", "sketch"]
def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", "-l", type=float, default= 0.01, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=31, help="Number of classes")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation size (between 0 and 1)")
    parser.add_argument("--source", choices=availabel_dataset, help="Training dataset", nargs='+')
    parser.add_argument("--target", choices=availabel_dataset, help="Test dataset" )
    parser.add_argument("--model", choices=["m2cl", 'resnet18'], default= "m2cl", help="Model to train")
    parser.add_argument("--saved_epoch", type=int, default= 20, help="Save model weight from this epoch")
    parser.add_argument("--checkpoint_path", "-p", type=str, default=None, help="Path to checkpoint")
    return parser.parse_args()
    

def M2CLTrainer(args):
    print('Using M2CL')
    network = M2CL18(args.n_classes, pretrained=True)
    optimizer = torch.optim.SGD(
            network.parameters(),
            lr=args.learning_rate,
            weight_decay=0.0005,
            momentum=0.9
        )
    
    #Load checkpoint
    if args.checkpoint_path:
        network.load_state_dict(torch.load(args.checkpoint_path))

    # #Get data
    # trainloader, valloader = get_train_dataloader(args.source,args.batch_size,args.val_size, augment_transform)
    # testloader = get_test_loader(args.target, args.batch_size)
    
    # for epoch in range(args.epochs):
    #     network.train()
    #     train_loss = 0
    #     true_pred = 0
    #     # Wrap trainloader with tqdm
    #     with tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch") as t:
    #         for x, y in t:
    #             preds, conv_act = network(x)
    #             y_tmp_np = y.cpu().detach().numpy()
    #             y_tmp = y_tmp_np.tolist()
    #             counts = {}
    #             same_indexes_tmp = {}
    #             dif_indexes = {}
    #             for i in y_tmp:
    #                 counts[i] = y_tmp.count(i)
    #                 same_indexes_tmp[i] = np.where(y_tmp_np == i)
    #                 dif_indexes[i] = np.where(y_tmp_np != i)

    #             same_indexes_tmp = OrderedDict(sorted(same_indexes_tmp.items()))
    #             same_indexes = []
    #             for i in range(len(same_indexes_tmp.items())):
    #                 if i in same_indexes_tmp.keys():
    #                     same_indexes.append(torch.combinations(torch.tensor(same_indexes_tmp[i][0])))

    #             custom_loss = my_loss(conv_act,
    #                                 same_indexes,
    #                                 0.01,
    #                                 1.0)

    #             ce_loss = F.cross_entropy(preds, y)

    #             loss = custom_loss + ce_loss
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #             train_loss += loss.detach()
    #             y_pred = torch.argmax(preds, 1)
    #             true_pred += torch.sum(y_pred == y).item()

    #             # Update tqdm description
    #             t.set_postfix(train_loss=train_loss.item() / len(trainloader.dataset),
    #                           accuracy=true_pred / len(trainloader.dataset))

    #     # Validation
    #     network.eval()
    #     val_loss_epoch = 0
    #     test_true_pred = 0
    #     for x, y in valloader:
    #         preds, conv_act = network(x)
    #         val_loss = F.cross_entropy(preds, y)
    #         val_loss_epoch += val_loss.detach()
    #         y_pred = torch.argmax(preds, 1)
    #         test_true_pred += torch.sum(y_pred == y).item()
    #     val_loss_epoch = val_loss_epoch / len(valloader.dataset)
    #     print(f"Validation loss: {val_loss_epoch}, accuracy: {test_true_pred/len(valloader.dataset)}")
    #     if epoch > args.saved_epoch:
    #         # checkpoint = {
    #         #     'model_state': network.state_dict(),
    #         #     'optimizer_state': optimizer.state_dict()
    #         # }
    #         # torch.save(checkpoint, f"checkpoint/m2cl_ckp_ep_{epoch}.pt")
    #         torch.save(network.state_dict(), f"checkpoint/m2cl_ckp_ep_{epoch}.pt")
    # test_loss = do_test(network, testloader)

def BaseRes18Trainer(args):
    print("Using Resnet 18")
    network = resnet18(pretrained=True, n_classes=args.n_classes)
    # print(network)
    optimizer = torch.optim.SGD(
        network.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0005,
        momentum=0.9
    )
    trainloader, valloader = get_train_dataloader(args.source, args.batch_size, args.val_size, augment_transform)
    testloader = get_test_loader(args.target, args.batch_size)
    for epoch in range(args.epochs):
        network.train()
        train_loss = 0
        true_pred = 0

        # Wrap trainloader with tqdm
        with tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch") as t:
            for x, y in t:
                network.train()
                y_pred = network(x)

                loss = F.cross_entropy(y_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.detach()
                prediction = torch.argmax(y_pred, 1)
                true_pred += torch.sum(prediction == y).item()

                # Update tqdm description
                t.set_postfix(train_loss=train_loss.item() / len(trainloader.dataset),
                              accuracy=true_pred / len(trainloader.dataset))

        print(f"Training loss at epoch {epoch}: {train_loss/len(trainloader.dataset)}, accuracy: {true_pred/len(trainloader.dataset)}")

        # Validation
        network.eval()
        val_loss_epoch = 0
        for x, y in valloader:
            preds = network(x)
            val_loss = F.cross_entropy(preds, y)
            val_loss_epoch += val_loss
        val_loss_epoch = val_loss_epoch / len(valloader.dataset)
        print(f"Validation loss: {val_loss_epoch}")
        del train_loss, loss, val_loss_epoch, val_loss
        if epoch > args.saved_epoch:
            torch.save(network.state_dict(), f"checkpoint/res18_ckp_ep_{epoch}.pt")
    test_loss = do_test_resnet(network, testloader)

def get_trainer(args):
    if args.model == "m2cl":
        M2CLTrainer(args)
    else:
        BaseRes18Trainer(args)
if __name__ == "__main__":
    args = get_args()
    get_trainer(args)
    
