# -*- coding: utf-8 -*-
"""
@Time ： 2023-04-28 22:31
@Auth ： xjjxhxgg
@File ：train.py
@IDE ：PyCharm
@Motto：xhxgg
"""

import torch.optim
from torch.utils.data import DataLoader


from model.DeepLabV3Plus import DeepLabV3Plus
from utils.trainer import one_epoch
from utils.Dataset import MyDataset

Epoch = 300
batch_size = 1
Init_lr = 0.00001
shuffle = True
num_workers = 0
Init_Epoch = 0

cuda = True
cross_entropy_weights=[1, 20, 20, 15, 5]
# model

local_data_path = 'Data/samples/all/'
local_save_path='../../out/models/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)
model = DeepLabV3Plus()
model.to(device)
model_train = model.train()
train_set = MyDataset(local_data_path, is_train=True, rate=1)
eval_set = MyDataset(local_data_path, is_train=False, rate=1)
train_gen = DataLoader(train_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                       drop_last=True, sampler=None)
eval_gen = DataLoader(eval_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                      drop_last=True, sampler=None)
# eval_gen=train_gen
optimizer = torch.optim.Adam(model.parameters(), lr=Init_lr)
history_acc = {'train': [], 'test': []}
epochs = []
if __name__ == '__main__':
    for epoch in range(Init_Epoch, Epoch):
        epoch_step_train = train_set.len // batch_size
        epoch_step_eval = eval_set.len // batch_size
        train_gen.dataset.epoch_now = epoch
        eval_gen.dataset.eppoch_now = epoch
        one_epoch(model_train=model_train, model=model, optimizer=optimizer, epoch=epoch, epoch_step=epoch_step_train,
                  epoch_step_val=epoch_step_eval, train_set=train_gen, eval_set=eval_gen, Epoch=Epoch, cuda=cuda,
                  hist_acc=history_acc,cross_entropy_weights=cross_entropy_weights,test=False)
        epochs.append(epoch + 1)
        number = epoch + 1
        name = str(number) if number >= 10 else f'0{number}'
        torch.save(model.state_dict(), local_save_path + f'{name}.mdl')
