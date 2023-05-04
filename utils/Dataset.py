# -*- coding: utf-8 -*-
"""
@Time ： 2023-04-24 2:33
@Auth ： xjjxhxgg
@File ：Dataset.py
@IDE ：PyCharm
@Motto：xhxgg
"""
import os

import cv2
import torchvision.transforms
from torch.utils.data.dataset import Dataset
class MyDataset(Dataset):
    def __init__(self, path, is_train, rate=0.9, max_h=224, max_w=224):
        '''
        :param path:
        :param is_train:
        :param rate: the proportion of train_set in total set
        '''
        super(MyDataset, self).__init__()
        self.max_w = max_w
        self.max_h = max_h
        self.path = path
        self.is_train = is_train
        self.rate = rate
        filenames = os.listdir(self.path)
        self.files = []
        self.masks = []
        for filename in filenames:
            if filename[-4:] == '.jpg':
                self.files.append(filename)
            else:
                self.masks.append(filename)
        self.masks.sort()
        self.files.sort()
        if is_train:
            self.files = self.files[:int(len(self.files) * rate)]
            self.masks = self.masks[:int(len(self.masks) * rate)]
        else:
            self.files = self.files[int(len(self.files) * rate):]
            self.masks = self.masks[int(len(self.masks) * rate):]
        self.len = len(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        image = cv2.imread(self.path + self.files[item])
        label = cv2.imread(self.path + self.masks[item])
        w, h = image.shape[1], image.shape[0]
        image = cv2.resize(image, (self.max_w, self.max_h))
        label = cv2.resize(label, (self.max_w, self.max_h)).transpose(2, 0, 1)
        label = label[0, :, :]
        transformer = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        image = transformer(image)
        image=image.detach().numpy()

        return image, label, self.masks[item], w, h