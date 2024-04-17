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
    


def get_random_subset(names, labels, percent):
    """

    :param names: list of names
    :param labels:  list of labels
    :param percent: 0 < float < 1
    :return:
    """
    samples = len(names)
    amount = int(samples * percent)
    random_index = sample(range(samples), amount)
    name_val = [names[k] for k in random_index]
    name_train = [v for k, v in enumerate(names) if k not in random_index]
    labels_val = [labels[k] for k in random_index]
    labels_train = [v for k, v in enumerate(labels) if k not in random_index]
    return name_train, name_val, labels_train, labels_val
  
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

def get_split_dataset_info(txt_list, val_percentage):
    names, labels = _dataset_info(txt_list)
    return get_random_subset(names, labels, val_percentage)


# def get_train_dataloader(source, batch_size,val_percentage, transform = augment_transform):
#     dataset_name = source
#     img_transformer = transform
#     name_train,name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % dataset_name), val_percentage)
#     train_dataset = MyDataset(name_train, labels_train, img_transformer=img_transformer)
#     val_dataset = MyDataset(name_val, labels_val, test_transform)
#     loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
#     return loader, val_loader
def get_train_dataloader(source, batch_size, val_percentage, transform = augment_transform):
    dataset_list = source
    assert isinstance(dataset_list, list)
    train_name_lst = []
    train_label_lst = []
    val_name_lst = []
    val_label_lst = []
    # val_datasets = []
    img_transformer = transform
    for dname in dataset_list:
        name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % dname), val_percentage)
        
        train_name_lst += (name_train)
        train_label_lst += (labels_train)
        val_name_lst+= (name_val)
        val_label_lst+=(labels_val)
    dataset = MyDataset(train_name_lst, train_label_lst, img_transformer)
    val_dataset = MyDataset(val_name_lst, val_label_lst, test_transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader

def get_test_loader(source, batch_size,transform = test_transform):
    dataset_name = source
    
    
    
    img_transformer = transform
    
    
    name_train, labels_train = _dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % dataset_name))
    train_dataset = MyDataset(name_train, labels_train, img_transformer=img_transformer)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    return loader
if __name__ == "__main__":
    
    train_dataloader, val_loader = get_train_dataloader(["amazon", "dslr", 'webcam'],4,0.2, augment_transform)
    print(len(train_dataloader.dataset))
    print(len(val_loader.dataset))
    

