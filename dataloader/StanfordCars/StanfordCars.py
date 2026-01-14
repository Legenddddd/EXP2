import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class StanfordCars(Dataset):

    def __init__(self, root='/data2/zlj/Datasets/', train=True,
                 index_path=None, index=None, base_sess=None):

        if train:
            setname = 'train'
            self.IMAGE_PATH = os.path.join(root, 'StanfordCars/cars_train')
        else:
            setname = 'test'
            self.IMAGE_PATH = os.path.join(root, 'StanfordCars/cars_test')
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        self.SPLIT_PATH = os.path.join(root, 'StanfordCars')

        csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data = []
        self.targets = []
        self.data2label = {}
        # lb = -1

        # self.wnids = []

        for l in lines:
            _, name, lb = l.split(',')
            path = osp.join(self.IMAGE_PATH, name)
            # if wnid not in self.wnids:
            #     self.wnids.append(wnid)
            #     lb += 1
            self.data.append(path)
            self.targets.append(int(lb)-1)
            self.data2label[path] = int(lb)-1


        if train:
            # image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                # transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            # self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            # image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)


    def SelectfromTxt(self, data2label, index_path):
        index = open(index_path).read().splitlines()
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.root, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets


if __name__ == '__main__':
    txt_path = "../../data/index_list/cub200/session_1.txt"
    # class_index = open(txt_path).read().splitlines()
    base_class = 100
    class_index = np.arange(base_class)
    dataroot = '/data2/zlj/Datasets/'
    batch_size_base = 400
    trainset = CUB200(root=dataroot, train=False,  index=class_index,
                      base_sess=True)
    cls = np.unique(trainset.targets)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
                                              pin_memory=True)

    # txt_path = "../../data/index_list/cifar100/session_2.txt"
    # # class_index = open(txt_path).read().splitlines()
    # class_index = np.arange(base_class)
    # trainset = CIFAR100(root=dataroot, train=True, download=True, transform=None, index=class_index,
    #                     base_sess=True)
    # cls = np.unique(trainset.targets)
    # trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
    #                                           pin_memory=True)
