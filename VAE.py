import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from torchvision.utils import save_image

# 自定义数据集类
class CustomImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.tif'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')  # 确保是RGB格式
        if self.transform:
            image = self.transform(image)
        return image, 0  # 返回0作为伪标签

# 修改VAE模型处理彩色图像
class ColorVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ColorVAE, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # [32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [64, 16, 16]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # [128, 8, 8]
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 潜在空间
        self.fc_mu = nn.Linear(128*8*8, latent_dim)
        self.fc_var = nn.Linear(128*8*8, latent_dim)
        
        # 解码器
        self.decoder_input = nn.Linear(latent_dim, 128*8*8)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [64, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [32, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),   # [3, 64, 64]
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder_input(z)
        return self.decoder(x)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 训练参数
latent_dim = 256
batch_size = 32
epochs = 40
image_size = 64   # 假设我们将所有图像调整为64x64

# 数据转换
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# 创建数据集和数据加载器
folder_path = "/tmp/readimg"  # 替换为你的图片文件夹路径
out_path = "/tmp/output"  # 替换为你的图片文件夹路径
dataset = CustomImageDataset(folder_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ColorVAE(latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 损失函数
def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 训练循环
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(dataloader.dataset):.4f}')

# 保存模型
#torch.save(model.state_dict(), 'color_vae.pth')

# 创建重建图像文件夹
os.makedirs('reconstructed_color_images', exist_ok=True)

# 生成并保存重建图像
model.eval()
with torch.no_grad():
    for i, (data, _) in enumerate(dataloader):
        data = data.to(device)
        recon_batch, _, _ = model(data)
        for j in range(recon_batch.size(0)):
            save_image(recon_batch[j], f'{out_path}/recon_{i*batch_size+j}.png')

print("所有彩色图片的重建结果已保存到 'reconstructed_color_images' 文件夹")
