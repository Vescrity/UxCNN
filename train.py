import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
from utils.data_loader import BasicDataset
import numpy as np
import matplotlib.pyplot as plt

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"

# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target.float()

        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output.squeeze(1), target)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch}, Progress: {batch_idx/len(train_loader)*100:.2f}%, Loss: {loss.item():.6f}')
    
    return running_loss / len(train_loader)

# 验证函数

def validate(model, device, val_loader, plot_distribution=False, plot_path=None):
    model.eval()
    running_loss = 0.0
    all_diffs = []  # 存储所有差异值
    cnt = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            target = target.float()

            output = model(data)
            diff = (output[0]-target[0])
            all_diffs.append(diff.item())  # 收集差异值
            
            if torch.abs(diff) < 0.5:
                cnt += 1
                print(f'[  {GREEN}OK{RESET}  ]', end='')
            else:
                print(f'[{RED}FAILED{RESET}]', end='')
            print(f'{output[0]}, {target[0]}')
            
            loss = F.mse_loss(output.squeeze(1), target)
            running_loss += loss.item()
    
    # 计算统计指标
    avg_loss = running_loss / len(val_loader)
    accuracy = cnt / len(val_loader)
    
    print(f"\nCorrect predictions: {cnt}/{len(val_loader)} ({accuracy:.2%})")
    print(f"Average loss: {avg_loss:.4f}")
    
    # 绘制并保存差异分布图
    if plot_distribution:
        plt.figure(figsize=(10, 6))
        
        # 绘制直方图
        plt.hist(all_diffs, bins=30, alpha=0.7, color='blue', edgecolor='black')
        
        # 添加统计信息
        mean_diff = np.mean(all_diffs)
        std_diff = np.std(all_diffs)
        plt.axvline(mean_diff, color='red', linestyle='dashed', linewidth=1)
        plt.axvline(mean_diff + std_diff, color='green', linestyle='dashed', linewidth=1)
        plt.axvline(mean_diff - std_diff, color='green', linestyle='dashed', linewidth=1)
        
        # 添加标签和标题
        plt.title('Prediction Error Distribution')
        plt.xlabel('Difference (Prediction - Target)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 添加图例
        plt.legend([
            f'Mean: {mean_diff:.4f}',
            f'Std: ±{std_diff:.4f}',
            'Error Distribution'
        ])
        
        # 保存图像到文件
        if plot_path is None:
            plot_path = "error_distribution.png"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，避免内存泄漏
        print(f"Error distribution plot saved to: {plot_path}")
    
    return avg_loss, accuracy


def train_model(
    model, device, 
    imgdirs,
    labels,
    epochs: int = 20,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent = 0.1,
    scale = 0.6
    ):
    
    dataset = BasicDataset(imgdirs, labels, scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    model = model.to(memory_format=torch.channels_last)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    ac = []
    for epoch in range(epochs):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        val_loss, acc = validate(model, device, val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        ac.append(acc)
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    return train_losses, val_losses, ac

def pltloss(title, train_losses, val_losses, ac, ylim = (0, 10)):
    plt.figure(figsize=(10, 5))
    num_epochs = len(train_losses)
    plt.plot(range(num_epochs), train_losses, label='Train Loss', marker='o')
    plt.plot(range(num_epochs), val_losses, label='Val Loss', marker='o')
    plt.plot(range(num_epochs), np.array(ac)*10, label='Accept', marker='+')
    plt.title(f'Loss of {title}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(ylim)
    plt.legend()
    plt.grid()
    plt.savefig(f'{title}.png')
    plt.close()
    
