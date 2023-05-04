# -*- coding: utf-8 -*-
"""
@Time ： 2023-04-28 21:51
@Auth ： xjjxhxgg
@File ：DeepLabV3Plus.py
@IDE ：PyCharm
@Motto：xhxgg
"""
import numpy as np
import torch
import torchvision.models as models
from torch import nn

# the model construction is not so difficult
# as long as you've learnt the basic structure
# of DeepLabV3+ model and ASPP,
# so I omit unnecessary detail descriptions about it

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=5):
        super(DeepLabV3Plus, self).__init__()
        # load the pretrained MobileNetV2 model as our DCNN
        self.backbone = models.mobilenet_v2(pretrained=True).features
        # declare an aspp attribution
        self.aspp = ASPP(in_channels=320, out_channels=256)
        self.decoder = Decoder(in_channels=256, out_channels=256, low_level_channels=24)
        self.last_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1)

    def __get_features(self, x):
        feature = []
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            feature.append(x)
        return feature

    def forward(self, x_):
        # Backbone
        input_size = x_.shape[2:]

        features = self.__get_features(x_)
        LLF = features[2]
        HLF = features[-2]
        # ASPP
        HLF = self.aspp(HLF)
        # Decoder
        merged = self.decoder(HLF, LLF)

        result = self.last_conv(merged)

        result = nn.functional.interpolate(result, size=input_size, mode='bilinear', align_corners=True)
        return result


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, multi_grid=None):
        super(ASPP, self).__init__()
        if multi_grid is None:
            multi_grid = [6, 12, 18]

        self.multi_grid = multi_grid
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.dilation_conv = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=self.multi_grid[0],
                      dilation=self.multi_grid[0]),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=self.multi_grid[1],
                      dilation=self.multi_grid[1]),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=self.multi_grid[2],
                      dilation=self.multi_grid[2]),

        ])
        self.image_pooling = nn.AdaptiveAvgPool2d(1)
        self.dim_reduce=nn.Conv2d(in_channels=320,out_channels=256,kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels * 5, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        out = []
        feature_map_size = x.shape[2:]
        x_ = self.conv1(x)
        out.append(x_)
        x_ = self.dilation_conv[0](x)
        out.append(x_)
        x_ = self.dilation_conv[1](x)
        out.append(x_)
        x_ = self.dilation_conv[2](x)
        out.append(x_)
        x_ = self.image_pooling(x)
        x_=self.dim_reduce(x_)
        x_ = nn.functional.interpolate(x_, size=feature_map_size, mode='bilinear', align_corners=True)
        out.append(x_)
        out = torch.cat(out, 1)
        out = self.conv2(out)
        return out



class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, low_level_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels + 48, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        x = nn.functional.interpolate(x, size=low_level_features.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        return x


if __name__ == '__main__':
    image = np.zeros((300, 5000, 3))
