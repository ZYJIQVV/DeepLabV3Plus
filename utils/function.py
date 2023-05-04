# -*- coding: utf-8 -*-
"""
@Time ： 2023-04-26 14:19
@Auth ： xjjxhxgg
@File ：function.py
@IDE ：PyCharm
@Motto：xhxgg
"""
import math

import cv2
import numpy as np
import torch
from torch import nn


def conv_block(in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1,
               dropout: int = 0):
    conv = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if dropout != 0:
        conv.append(nn.Dropout(dropout))
    return conv


def block(in_channels: int, out_channels: list = 3, kerner_size: list = None,
          stride=None,
          padding=None, dilate=None):
    if dilate is None:
        dilate = [1, 2, 1]
    if padding is None:
        padding = [0, 0, 0]
    if stride is None:
        stride = [1, 1, 1]
    if kerner_size is None:
        kerner_size = [1, 3, 1]

    conv = nn.Sequential(
        *conv_block(in_channels=in_channels, out_channels=out_channels[0], kernel_size=kerner_size[0],
                    stride=stride[0], padding=padding[0], dilation=dilate[0]),
        *conv_block(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=kerner_size[1],
                    stride=stride[1], padding=padding[1], dilation=dilate[1]),
        *conv_block(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=kerner_size[2],
                    stride=stride[2], padding=padding[2], dilation=dilate[2])
    )
    return conv


def conv_(in_channels, out_channels, kernel_size, stride=1, padding=0, rate: int = 1):
    return nn.Sequential(
        *conv_block(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=rate)).cuda()


def mask_to_rgb(mask):
    '''
    :param mask: single channel mask
    :return:
    '''
    temp_mask=mask.transpose(1,2,0)
    color = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 255)}
    temp = np.zeros((temp_mask.shape[0], temp_mask.shape[1], 3))
    temp[:, ] = temp_mask
    for i in range(temp_mask.shape[0]):
        for j in range(temp_mask.shape[1]):
            temp[i, j] = color[temp_mask[i, j, 0]]

    return temp.astype(np.uint8)


# 得到混淆矩阵
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


# 计算图像分割衡量系数
def label_accuracy_score(label_trues, label_preds, n_class):
    """
     :param label_preds: numpy data, shape:[batch,h,w]
     :param label_trues:同上
     :param n_class:类别数
     Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IOU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()

    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iou = np.nanmean(iou)

    freq = hist.sum(axis=1) / hist.sum()
    fwiou = (freq[freq > 0] * iou[freq > 0]).sum()
    return acc, acc_cls, mean_iou, fwiou


# these 2 functions are used to match the ji.py file
# due to the exceeding size of the input_image given to function process_image
# the model can't be directly applied on the input_image
# the image has to be resized to meet the max size of the model
# after the model solve the mask, the mask has the size which was the one of the resized input_image
# and doesnot match the size of the label
# so the output mask has to be re-resized back to the original size
def input_resize(image):
    # max_w,max_h=1920,1080
    max_w, max_h = 224, 224
    h, w, _ = image.shape
    if h > max_h and w > max_w:
        image = cv2.resize(image, (max_h, max_w)).transpose(2, 0, 1).astype(np.float32)
    elif h > 1080:
        image = cv2.resize(image, (max_h, w)).transpose(2, 0, 1).astype(np.float32)
    elif w > 1920:
        image = cv2.resize(image, (h, max_w)).transpose(2, 0, 1).astype(np.float32)
    else:
        image = image.transpose(2, 0, 1).astype(np.float32)
    return image


def output_resize(image, w, h):
    image = image.transpose(1, 2, 0).astype(np.int8)
    output = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST).reshape((h, w, 1))
    return output


# # get mask from the output of the model
# def get_mask(prob):
#

def read(path):
    image = cv2.imread(path)
    # w, h = image.shape[1], image.shape[0]
    h, w, c = image.shape
    # max_w,max_h=1920,1080
    # max_w, max_h = 224, 224
    max_w, max_h = 6000, 6000
    # print(h,w)
    if h > max_h and w > max_w:
        # if h > 5000 or w > 5000:
        image = cv2.resize(image, (max_h, max_w)).transpose(2, 0, 1)
    elif h > max_h:
        image = cv2.resize(image, (max_h, w)).transpose(2, 0, 1)
    elif w > max_w:
        image = cv2.resize(image, (h, max_w)).transpose(2, 0, 1)
    else:
        image = image.transpose(2, 0, 1)
    return torch.from_numpy(image).unsqueeze(0).type(torch.float32).cuda()


# this function first check if the size can be divided by 8
# if not it will fill zeros to make the size divisible by 8
# but this will require a deletion operation on the output
# mask to eliminate the elements that have been added before
# and resize the image back to the original size
# then, it will check if the size is over [1080,1920]
# if true, it will crop the image into 4 parts
# so that the size doesn't exceed [1080,1920]
# but on occasion of this cropping
# I'll have concat the 4 parts together back to a full image
# and notice that after this cropping, some part of the image
# may not be dividable by 8,
# so it requires another fill
# to cut down the operations
# I first apply the crop operation before check
# if the image is dividable by 8
def crop(image):
    '''
    in this function we crop image into acceptable size,
    the sub-image from cropping may not reach the required size,
    so we put this sub-image at upper left and fill with 0 at other pixels

    in order to facilitate merging, I set cropped image set as a list A,
    the image is cropped into m part in width and n in height,
    that is cropped into n whole rows and m whole cols
    so A has n elements and each represents a whole row
    for a certain element B in A, which represents a certain whole row,
    there are m elements in B and each represents a whole col in the local whole row B
    :param image:[C,H,W]
    :return: cropped image list,crop_num_h,crop_num_w
    '''
    max_h = 352
    max_w = 352
    c, h, w = image.shape[-3],image.shape[-2],image.shape[-1]
    if h <= max_h and w <= max_w and h % 16 == 0 and w % 16 == 0:
        '''no need to crop, just pad with 0'''
        result = np.zeros((c, max_h, max_w))
        result[:, :h, :w] = image
        return [[result]],0,0
    if h <= max_h and w > max_w and h % 16 == 0:
        '''no need to crop on height, need to crop on width'''
        # crop_num represents the number of sub-image to be cropped
        # it is generated from:
        crop_num = math.ceil(w / max_w)
        divisible = True if w % max_w == 0 else False
        target_h, target_w = h, crop_num * max_w
        padded_image = pad(image, w, h, target_w, target_h,c)
        result=np.split(padded_image,crop_num,axis=2)

        return [result],0,crop_num
    if h <= max_h and w > max_w and h % 16 != 0:
        '''no need to crop on height, need to crop on width'''
        # crop_num represents the number of sub-image to be cropped
        # it is generated from:
        crop_num = math.ceil(w / max_w)
        divisible = True if w % max_w == 0 else False
        target_h, target_w = math.ceil(h/16)*16, crop_num * max_w
        padded_image = pad(image, w, h, target_w, target_h,c)
        result=np.split(padded_image,crop_num,axis=2)

        return [result],0,crop_num
    if h>max_h and w<=max_w and w % 16 == 0:
        '''no need to crop on width, need to crop on height'''
        crop_num=math.ceil(h/max_h)
        target_h,target_w=crop_num*max_h,w
        padded_image=pad(image,w,h,target_w,target_h,c)
        temp=np.split(padded_image,crop_num,axis=1)
        # here, each element in temp represents a whole col, which is violent to
        # what I set at first,
        # so I need to reform temp
        result = []
        for t in temp:
            result.append([t])
        return result,crop_num,0

    if h>max_h and w<=max_w and w % 16 != 0:
        '''no need to crop on width, need to crop on height'''
        crop_num=math.ceil(h/max_h)
        target_h,target_w=crop_num*max_h,math.ceil(w/16)*16
        padded_image=pad(image,w,h,target_w,target_h,c)
        temp=np.split(padded_image,crop_num,axis=1)
        # here, each element in temp represents a whole col, which is violent to
        # what I set at first,
        # so I need to reform temp
        result = []
        for t in temp:
            result.append([t])
        return result,crop_num,0
    if h>max_h and w>max_w:
        '''need to crop on both width and height'''
        crop_num_w=math.ceil(w/max_w)
        crop_num_h=math.ceil(h/max_h)
        target_h,target_w=crop_num_h*max_h,crop_num_w*max_w
        padded_image=pad(image,w,h,target_w,target_h,c)
        '''crop on height first'''
        result=[]
        cropped_on_h=np.split(padded_image,crop_num_h,axis=1)
        for h in cropped_on_h:
            temp=np.split(h,crop_num_w,axis=2)
            result.append([*temp])
        return result,crop_num_h,crop_num_w
    # not divisible
    if h % 16 != 0 and w % 16 == 0:
        target_h=math.ceil(h/16)*16
        result=pad(image,w,h,w,target_h,c)
        return [[result]],0,0
    if h % 16 == 0 and w % 16 != 0:
        target_w=math.ceil(w/16)*16
        result=pad(image,w,h,target_w,h,c)
        return [[result]],0,0
    if h % 16 != 0 and w % 16 != 0:
        target_h = math.ceil(h / 16)*16
        target_w = math.ceil(w / 16)*16
        result = pad(image, w, h, target_w, target_h, c)
        return [[result]], 0, 0



def pad(array, w, h, tw, th,channel):
    '''
    :param array:
    :param w: original w
    :param h: original h
    :param tw: target w
    :param th: target h
    :return:
    '''
    result = np.zeros((channel, th, tw))
    result[:, :h, :w] = array
    return result

def merge(array_list,channel):
    w = array_list[0][0].shape[2]
    h = array_list[0][0].shape[1]
    total_w=w*len(array_list[0])
    total_h=h*len(array_list)
    result=np.zeros((channel,total_h,total_w))
    CNT=0
    for i in array_list:
        CNT+=1
        cnt=0
        temp=np.zeros((channel,h,total_w))
        for j in i:
            cnt +=1
            temp[:,:,int((cnt-1)*w):int(cnt*w)]=j
        result[:,int((CNT-1)*h):int(CNT*h),:]=temp
    return result.astype(np.uint8)

def random_crop(image,max_h=224,max_w=224):
    c,h,w=image.shape
    if h<=max_h and w <=max_w:
        return image

    if h<=max_h and w>max_w:
        difference_w = w - max_w
        start_w = np.random.randint(0, difference_w)
        out = image[:, :, start_w:int(start_w + max_w)]
        return out
    if h>max_h and w<=max_w:
        difference_h = h - max_h
        start_h = np.random.randint(0, difference_h)
        out = image[:, start_h:int(start_h + max_h), :]
        return out
    if h>max_h and w>max_w:
        difference_h = h - max_h
        difference_w = w - max_w
        start_h = np.random.randint(0, difference_h)
        start_w = np.random.randint(0, difference_w)
        out = image[:, start_h:int(start_h + max_h), start_w:int(start_w + max_w)]
        return out