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

def train_and_plot(net, title, epochs=20):
    tloss, vloss = train_model(net, device, imgdirs, labels, epochs)
    pltloss(title, tloss, vloss)
    
from CNN import CNN2D
train_and_plot(CNN2D(1), 'CNN', epochs=50)
train_and_plot(UsCNN(), 'UsCNN')
train_and_plot(UrCNN(), 'UrCNN')
train_and_plot(UscCNN(), 'UscCNN')
