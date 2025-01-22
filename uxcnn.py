import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from unet import UNet

import os
import glob
import random
import pandas as pd
from PIL import Image
import numpy as np
TARGET_LENGTH = 8192


def extract_sorted(color_image: torch.Tensor, binary_image: torch.Tensor) -> torch.Tensor:
    # 调整 color_image 的分辨率为 binary_image 的大小
    color_image_resized = F.interpolate(color_image.unsqueeze(0), size=binary_image.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
    # 提取白色像素
    white_pixels = binary_image > 0.5
    extracted_pixels = color_image[white_pixels]

    # 计算亮度（灰度值），使用平均值
    brightness = extracted_pixels.mean(dim=1)

    # 获取排序的索引
    sorted_indices = torch.argsort(brightness)
    sorted_pixels = extracted_pixels[sorted_indices]

    return sorted_pixels


def extract_randomized(color_image: torch.Tensor, binary_image: torch.Tensor) -> torch.Tensor:
    # 调整 color_image 的分辨率为 binary_image 的大小
    color_image_resized = F.interpolate(color_image.unsqueeze(0), size=binary_image.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
    # 提取白色像素
    white_pixels = binary_image > 0.5
    extracted_pixels = color_image[white_pixels]

    # 随机打乱提取的像素
    indices = torch.randperm(extracted_pixels.size(0))
    shuffled_pixels = extracted_pixels[indices]

    return shuffled_pixels


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=3, padding=1)  # 输入通道为 3，RGB
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * (TARGET_LENGTH // 4), 128)  # 根据输入形状调整
        self.fc2 = nn.Linear(128, 1)  # 输出一个浮点数

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * (TARGET_LENGTH // 4))  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def unet_preprocess(img):
    pass
    return img

def unet_afterprocess(img):
    pass
    return img

class UxCNN(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 1):
        super(UxCNN, self).__init__()
        self.unet = UNet(n_channels, n_classes)
        self.cnn = SimpleCNN()
    def xfunc(self, binary):
        raise NotImplementedError("Subclasses must implement this method")
        return binary
    
        
    def forward(self, x):
        ori_img = x
        uin = unet_preprocess(x)
        # UNet 输出二值化图像
        uout = self.unet(uin)
        binary_image = unet_afterprocess(uout)
        st = xfunc(ori_img, binary_image)
        if len(st) > TARGET_LENGTH:
            indices = torch.linspace(0, len(st) - 1, TARGET_LENGTH).long().to(x.device)
            resized_pixels = st[indices]
        elif len(st) < TARGET_LENGTH:
            resized_pixels = torch.nn.functional.pad(st, (0, 0, 0, TARGET_LENGTH - len(st)), mode='edge')
        else:
            resized_pixels = st

        output = self.cnn(resized_pixels)
        return output
    
class UsCNN(UxCNN):
    def xfunc(self, ori_img ,binary):
        return extract_sorted(ori_img, binary)
    
class UrCNN(UxCNN):
    def xfunc(self, ori_img, binary):
        return extract_randomized(ori_img, binary)
    
