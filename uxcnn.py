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

def extract_white_pixels(color_image: torch.Tensor, binary_image: torch.Tensor) -> torch.Tensor:
    assert color_image.dim() == 4 and binary_image.dim() == 4, "Both inputs must be 4D tensors"
    white_pixels_mask = binary_image > 0.5  # 形状为 (1, 1, H, W)

    # 扩展掩码以匹配 color_image 的形状
    white_pixels_mask = white_pixels_mask.expand(-1, 3, -1, -1)  # 形状变为 (1, 3, H, W)
    extracted_pixels = color_image[white_pixels_mask].view(-1, 3)  # 形状为 (num_white_pixels, 3)

    return extracted_pixels

def extract_sorted(color_image: torch.Tensor, binary_image: torch.Tensor) -> torch.Tensor:

    extracted_pixels = extract_white_pixels(color_image, binary_image)
    brightness = extracted_pixels.mean(dim=1)
    sorted_indices = torch.argsort(brightness)
    sorted_pixels = extracted_pixels[sorted_indices]
    output = sorted_pixels.unsqueeze(0).permute(0, 2, 1)  # 变换为 (1, 3, n)
    return output

def extract_sorted_with_center(color_image: torch.Tensor, binary_image: torch.Tensor) -> torch.Tensor:
    extracted_pixels = extract_white_pixels(color_image, binary_image)
    brightness = extracted_pixels.mean(dim=1)  # 计算亮度
    sorted_indices = torch.argsort(brightness)  # 按照亮度升序排序
    sorted_pixels = extracted_pixels[sorted_indices]

    # 分离下标为偶数和奇数的像素
    even_indices = sorted_pixels[::2]  # 下标为偶数的像素
    odd_indices = torch.flip(sorted_pixels[1::2], dims = [0])  # 下标为奇数的像素，倒序

    # 拼接
    output = torch.cat((even_indices, odd_indices), dim=0)

    output = output.unsqueeze(0).permute(0, 2, 1)  # 变换为 (1, 3, n)
    return output

def extract_randomized(color_image: torch.Tensor, binary_image: torch.Tensor) -> torch.Tensor:
    extracted_pixels = extract_white_pixels(color_image, binary_image)
    n = extracted_pixels.shape[0]  
    if n == 0:
        return torch.empty((1, 3, 0)) 
    indices = torch.randperm(n) 
    random_pixels = extracted_pixels[indices]
    output = random_pixels.unsqueeze(0)  # 变换为 (1, n, 3)
    output = output.permute(0, 2, 1)  # 变换为 (1, 3, n)
    return output



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
    @staticmethod
    def xfunc(color_image ,binary):
        raise NotImplementedError("Subclasses must implement this method")
        return binary
    
        
    def forward(self, x):
        ori_img = x
        uin = unet_preprocess(x)
        # UNet 输出二值化图像
        uout = self.unet(uin)
        binary_image = unet_afterprocess(uout)
        st = self.xfunc(ori_img, binary_image)
        resized_pixels = F.interpolate(st, size=(TARGET_LENGTH,), mode='linear', align_corners=False)
        output = self.cnn(resized_pixels)
        return output
    
class UsCNN(UxCNN):
    @staticmethod
    def xfunc(ori_img ,binary):
        return extract_sorted(ori_img, binary)
    
class UrCNN(UxCNN):
    @staticmethod
    def xfunc(ori_img, binary):
        return extract_randomized(ori_img, binary)
    
class UscCNN(UxCNN):
    @staticmethod
    def xfunc(ori_img, binary):
        return extract_sorted_with_center(ori_img, binary)
