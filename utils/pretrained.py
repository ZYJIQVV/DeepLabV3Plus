# -*- coding: utf-8 -*-
"""
@Time ： 2023-04-19 21:39
@Auth ： xjjxhxgg
@File ：pretrained.py
@IDE ：PyCharm
@Motto：xhxgg
"""

import torch
import torchvision.models
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pretrained_models(model):
    if model == 'resnet50':
        ResNet50 = models.resnet50(pretrained=True)
        ResNet50.to(device)
        ResNet50.eval()
        return ResNet50
    if model == 'resnet101':
        ResNet101 = models.resnet101(pretrained=True)
        ResNet101.to(device)
        ResNet101.eval()
        return ResNet101
    if model == 'resnet34':
        ResNet34 = models.resnet34(pretrained=True)
        ResNet34.to(device)
        ResNet34.eval()
        return ResNet34
    if model == 'vgg16':
        VGG16 = models.vgg16(pretrained=True)
        VGG16.to(device)
        VGG16.eval()
        return VGG16
    if model == 'vgg19':
        VGG19 = models.vgg19(pretrained=True)
        VGG19.to(device)
        VGG19.eval()
        return VGG19
