import numpy as np
from uxcnn import UsCNN, UrCNN, UscCNN
from train import train_model, pltloss, torch


imgdirs = [ f'data/{i}' for i in range(6, 13) ]
labels = np.arange(-6.0, -13.0, -1.0, dtype=np.float32)

scale = 0.6
val_percent = 0.1
batch_size = 1
learning_rate = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tloss, vloss = train_model(UscCNN(), device, imgdirs, labels)
pltloss('usccnn.png', tloss, vloss)


