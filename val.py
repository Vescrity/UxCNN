import numpy as np
import os
from uxcnn import UsCNN, UrCNN, UscCNN, CVsCNN
from train import train_model, pltloss, torch
from torch.utils.data import DataLoader, random_split
from utils.data_loader import BasicDataset

imgdirs = [ f'data/{i}' for i in range(6, 16) ]
labels = np.arange(-6.0, -16.0, -1.0, dtype=np.float32)

scale = 0.6
val_percent = 0.9
batch_size = 1
learning_rate = 7e-6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modpath = 'm.pth'
    
from CNN import CNN2D
from Myvit import vit
dataset = BasicDataset(imgdirs, labels, scale)
loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
all_loader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_args)


mod = vit
mod.load_state_dict(torch.load(modpath, map_location=device))
from train import validate
validate(mod, device, all_loader)

