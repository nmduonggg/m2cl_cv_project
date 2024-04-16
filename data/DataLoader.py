import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import sample, random
from os.path import join, dirname

augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels




class MyDataset(data.Dataset):
    def __init__(self, names, labels,  img_transformer=test_transform):
        self.data_path = ""
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer
    
    def get_image(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img)
        
    def __getitem__(self, index):
        img = self.get_image(index)
        return img, int(self.labels[index])

        
    def __len__(self):
        return len(self.names)
    
def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels

def get_train_dataloader(source, batch_size, transform = augment_transform):
    dataset_name = source
    
    
    
    img_transformer = transform
    
    
    name_train, labels_train = _dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % dataset_name))
    train_dataset = MyDataset(name_train, labels_train, img_transformer=img_transformer)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    return loader

def get_test_loader(source, batch_size,transform = test_transform):
    dataset_name = source
    
    
    
    img_transformer = transform
    
    
    name_train, labels_train = _dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % dataset_name))
    train_dataset = MyDataset(name_train, labels_train, img_transformer=img_transformer)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    return loader
if __name__ == "__main__":
    
    dataloader = get_train_dataloader("amazon", augment_transform, 4)
    print(len(dataloader))

