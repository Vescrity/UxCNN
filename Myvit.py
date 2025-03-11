import torch
import torch.nn as nn
from torchvision.models import vit_b_16

class MyViT(nn.Module):
    def __init__(self, model):
        super(MyViT, self).__init__()
        self.model = model
        self.pool = nn.AdaptiveAvgPool2d((224, 224))  # 调整输出大小为 224x224

    def forward(self, x):
        x = self.pool(x)  # 调整输入图像大小
        return self.model(x)

vit = MyViT(vit_b_16(pretrained=True))

# 替换最后一层
num_features = vit.model.heads.head.in_features
vit.model.heads.head = nn.Linear(num_features, 1) 
