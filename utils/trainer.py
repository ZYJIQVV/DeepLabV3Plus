# -*- coding: utf-8 -*-
"""
@Time ： 2023-04-28 22:31
@Auth ： xjjxhxgg
@File ：trainer.py
@IDE ：PyCharm
@Motto：xhxgg
"""
# -*- coding: utf-8 -*-

import os

import numpy as np
import torch
from tqdm import tqdm

from .function import *


def one_epoch(model_train, model, train_set, eval_set, optimizer, cuda, epoch_step, epoch, Epoch, epoch_step_val=None,
              save=False, filename=None,
              hist_acc: dict = None, cross_entropy_weights=None,test=False):
    if cross_entropy_weights is None:
        cross_entropy_weights = [5, 5, 6, 15, 3]
    loss = 0
    acc = 0
    mean_iou = 0
    fwiou = 0
    print('train')
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(cross_entropy_weights))
    pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)  # 进度条对象
    if cuda:
        criterion = criterion.cuda()
    for iter, batch in enumerate(train_set):
        model.train()
        if iter >= epoch_step:
            break
        # these paths are used for saving original images, labels and masks to facilitate observation
        image, label, image_name, w, h = batch[0], batch[1], batch[2][0], batch[3][0], batch[4][0]
        image_save_path = 'E:/Homework/PatternRecognition/Labs/FinalWork/DeepLab/3/out/image/'
        label_save_path = 'E:/Homework/PatternRecognition/Labs/FinalWork/DeepLab/3/out/label/'
        mask_save_path = 'E:/Homework/PatternRecognition/Labs/FinalWork/DeepLab/3/out/mask/'
        jpg_path = 'E:/Homework/PatternRecognition/Labs/FinalWork/DeepLab/3/train/src_repo/Data/samples/jpg/'
        png_path = 'E:/Homework/PatternRecognition/Labs/FinalWork/DeepLab/3/train/src_repo/Data/samples/png/'
        image_to_save = cv2.imread(jpg_path + image_name[:-4] + '.jpg')
        label_to_save=cv2.imread(png_path+image_name[:-4] + '.png',cv2.IMREAD_GRAYSCALE)
        label_to_save=np.reshape(label_to_save,(label_to_save.shape[0],label_to_save.shape[1],1)).transpose(2,0,1)
        cv2.imwrite(image_save_path + image_name, image_to_save)
        cv2.imwrite(label_save_path+image_name,mask_to_rgb(label_to_save))

        with torch.no_grad():
            if cuda:
                image = image.cuda()
                label = label.cuda()
        label = label.long()
        model_train.train()
        prob = model_train(image).type(torch.float32)
        prob = torch.softmax(prob, dim=1)
        loss_value = criterion(prob, label)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        loss += loss_value.item()
        # below are not so important parts,
        # they are for transforming the prediction into colored mask for observation
        prob_tensor = torch.argmax(prob, dim=1)
        prob_tensor = prob_tensor.cuda().type(torch.float32)
        color_mask = mask_to_rgb(prob_tensor.cpu().detach().numpy())

        # because w and h got from dataset will be turned into a tensor,
        # in order to resize the image using w and h we need to get their values as an integer
        if isinstance(w,torch.Tensor):
            w=w.item()
            h=h.item()
        color_mask = cv2.resize(color_mask, (w, h))
        cv2.imwrite(mask_save_path + image_name, color_mask)

        # label_accuracy_score is a function calculating the iou,
        # but it seems there are some bugs with it,
        # I'll take time to amend it later,
        # just ignore it at this phase
        acc_temp, acc_cls, mean_iou_temp, fwiou_temp = label_accuracy_score(
            label_trues=label.type(torch.uint8).cpu().detach().numpy(),
            label_preds=np.resize(prob_tensor.unsqueeze(0).type(torch.uint8).cpu().detach().numpy(), (1,256, 256)), n_class=5)
        acc += acc_temp
        acc /= (iter + 1)
        mean_iou += mean_iou_temp
        mean_iou /= (iter + 1)
        fwiou += fwiou_temp
        fwiou /= (iter + 1)

        pbar.set_postfix(**{'loss': loss / (iter + 1), 'acc': acc, 'mean_iou': mean_iou, 'w_iou': fwiou})
        pbar.update(1)
    if hist_acc is not None:
        hist_acc['train'].append(acc)
    if test==False:
        return
    print('test')
    acc = 0
    mean_iou = 0
    fwiou = 0
    pbar.close()
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    acc = 0
    for iter, batch in enumerate(eval_set):
        if iter >= epoch_step_val:
            break
        model_train.eval()
        image, label, image_name, w, h = batch[0], batch[1], batch[2][0], batch[3][0], batch[4][0]
        image_save_path = 'E:/Homework/PatternRecognition/Labs/FinalWork/DeepLab/3/out/image/'
        label_save_path = 'E:/Homework/PatternRecognition/Labs/FinalWork/DeepLab/3/out/label/'
        mask_save_path = 'E:/Homework/PatternRecognition/Labs/FinalWork/DeepLab/3/out/mask/'
        jpg_path = 'E:/Homework/PatternRecognition/Labs/FinalWork/DeepLab/3/train/src_repo/Data/samples/jpg/'
        png_path = 'E:/Homework/PatternRecognition/Labs/FinalWork/DeepLab/3/train/src_repo/Data/samples/png/'
        image_to_save = cv2.imread(jpg_path + image_name[:-4] + '.jpg')
        label_to_save = cv2.imread(png_path + image_name[:-4] + '.png', cv2.IMREAD_GRAYSCALE)
        label_to_save = np.reshape(label_to_save, (label_to_save.shape[0], label_to_save.shape[1], 1)).transpose(2, 0, 1)
        cv2.imwrite(image_save_path + image_name, image_to_save)
        cv2.imwrite(label_save_path + image_name, mask_to_rgb(label_to_save))
        with torch.no_grad():
            if cuda:
                image = image.cuda()
                label = label.cuda()
        label = label.long()
        model_train.train()
        prob = model_train(image).type(torch.float32)
        prob = torch.softmax(prob, dim=1)
        prob_tensor = torch.argmax(prob, dim=1)
        prob_tensor = prob_tensor.cuda().type(torch.float32)
        color_mask = mask_to_rgb(prob_tensor.cpu().detach().numpy())
        if isinstance(w,torch.Tensor):
            w=w.item()
            h=h.item()
        color_mask = cv2.resize(color_mask, (w, h))

        cv2.imwrite(mask_save_path + image_name, color_mask)

        acc_temp, acc_cls, mean_iou_temp, fwiou_temp = label_accuracy_score(
            label_trues=label.type(torch.int).cpu().detach().numpy(),
            label_preds=np.resize(prob_tensor.unsqueeze(0).type(torch.uint8).cpu().detach().numpy(), (1,256, 256)), n_class=5)
        acc += acc_temp
        acc /= (iter + 1)
        mean_iou += mean_iou_temp
        mean_iou /= (iter + 1)
        fwiou += fwiou_temp
        fwiou /= (iter + 1)
        pbar.set_postfix(**{'acc': acc, 'mean_iou': mean_iou, 'fwiou': fwiou})
        pbar.update(1)
    if hist_acc is not None:
        hist_acc['train'].append(acc)

    pbar.close()
    if save:
        if filename is None:
            raise 'To save the model, you need assign the \'filename\' parameter'
        else:
            torch.save(model.state_dict(), os.path.join(filename, 'model_last.pt'))
