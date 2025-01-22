import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
class BasicDataset(Dataset):
    def __init__(self, imgdirs, labels, scale: float = 1.0):
        self.image_paths = []
        self.labels = []
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        # 加载数据
        for data_dir, value in zip(imgdirs, labels):
            images = glob.glob(os.path.join(data_dir, '*.tif')) 
            for img in images:
                self.image_paths.append(img)
                self.labels.append(value)
        

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def preprocess(pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample= Image.BICUBIC)
        img = np.asarray(pil_img)

        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))

        if (img > 1).any():
            img = img / 255.0

        return img

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path)
        image = self.preprocess(image, self.scale)
        
        return torch.as_tensor(image.copy()).float().contiguous(), label


