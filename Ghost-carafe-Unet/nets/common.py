import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class GhostConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        # self.cv2 = Conv(c_, c_ // 2, 5, 1, None, c_ // 2, act=act)
        # self.cv3 = Conv(c_, c_ // 2, 7, 1, None, c_ // 2, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)
    def forward(self, x):
        y = self.cv1(x)
        # return torch.cat((y, self.cv2(y), self.cv3(y)), 1)
        return torch.cat((y, self.cv2(y)), 1)


# 定制化
class CARAFE(nn.Module):
    # 参数： 输入通道c1  输出通道c2  
    def __init__(self, c1, c2, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        # 定义卷积层构建模型
        self.down = GhostConv(c1, c1 // 4, 1)
        # self.down = nn.Conv2d(c1, c1 // 4, 1)
        self.encoder = GhostConv(c1 // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        # self.encoder = nn.Conv2d(c1 // 4, self.up_factor ** 2 * self.kernel_size ** 2,
        #                         self.kernel_size, 1, self.kernel_size // 2)
        self.out = GhostConv(c1, c2, 1)
        # self.out = nn.Conv2d(c1, c2, 1)
    # 前向传播方法
    # 参数： 输入张量X, 返回输出结果X.
    def forward(self, x):
        N, C, H, W = x.size()
        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        # 降维
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        # 通过encoder进行卷积,调整通道为up_factor^2,同时将图像尺寸放大up_factor倍.
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        # softmax
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        # upfold 对卷积结果进行展开操作,然后根据up_factor进行滑动窗口切片.
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        # 调整形状,为了后续的矩阵乘法
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W,
                                              self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        # 对张量的维度进行转置,使得卷积核的维度排列在最后两个维度
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)
        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        # 填充
        x = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2,
                          self.kernel_size // 2, self.kernel_size // 2),
                  mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        # unfold对填充后的结果进行展开,按照kernel_size进行滑动窗口切片.
        x = x.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        x = x.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        # 形状调整
        x = x.reshape(N, C, H, W, -1)  # (N, C, H, W, Kup^2)
        # 维度转置
        x = x.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)
        # 矩阵乘法,得到特征图
        out_tensor = torch.matmul(x, kernel_tensor)  # (N, H, W, C, S^2)
        # 调整形状
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        # 像素混洗,此时特征图缩小为原来的up_factor倍
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        # 最后卷积,得到特征图
        out_tensor = self.out(out_tensor)
        # print("up shape:",out_tensor.shape)
        return out_tensor
