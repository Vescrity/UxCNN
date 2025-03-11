import numpy as np
from uxcnn import UsCNN, UrCNN, UscCNN, CVsCNN
from train import train_model, pltloss, torch

imgdirs = [ f'data/{i}' for i in range(6, 13) ]
labels = np.arange(-6.0, -13.0, -1.0, dtype=np.float32)

scale = 0.6
val_percent = 0.2
batch_size = 1
learning_rate = 7e-6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_plot(net, title, epochs=40):
    tloss, vloss,ac = train_model(net, device, imgdirs, labels, epochs,batch_size, learning_rate, val_percent)
    pltloss(title, tloss, vloss, ac)
    
from CNN import CNN2D
from Myvit import vit
#train_and_plot(vit, 'Vit', epochs=20)
#train_and_plot(CNN2D(1), 'CNN', epochs=100)
#train_and_plot(UsCNN(), 'UsCNN')
#train_and_plot(UrCNN(), 'UrCNN')
#train_and_plot(UscCNN(), 'UscCNN')
#
train_and_plot(CVsCNN(),'cvcnn')
