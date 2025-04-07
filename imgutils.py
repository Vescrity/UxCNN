import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

SKIP_SHOW = False
SAVE_COMPARE = False
GET_LIST = False
class Droplet:
    def __init__(self, img, brightness, area):
        self.img = img
        self.brightness = brightness
        self.area = area

def show_image(image_data):
    """
    显示图片数据，支持灰度图和彩色图。
    
    参数:
    image_data (numpy.ndarray): 输入的图片数据，可以是灰度或彩色。
    """
    if SKIP_SHOW:
        return None
    # 判断图像的维度
    if len(image_data.shape) == 2:  # 灰度图
        plt.imshow(image_data, cmap='gray')
    elif len(image_data.shape) == 3:  # 彩色图
        plt.imshow(image_data)
    else:
        raise ValueError("输入的图像数据格式不正确。")

    plt.axis('off')  # 关闭坐标轴
    plt.show()  # 展示图像

import torch

def tensor2im(tensor):
    """
    将 PyTorch 张量转换为彩色图像。
    """
    tensor = tensor.cpu()
    image = tensor[0].permute(1, 2, 0).numpy() * 255.0
    return image.astype(np.uint8)

def im2tensor(image):
    """
    将通过 cv2.imread 读取的彩色图像转换为 PyTorch 张量。
    """
    # 将图像的形状从 (H, W, C) 转换为 (C, H, W)
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return image_tensor


def bin2tensor(image):

    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 将二值化图像转换为张量
    # 将图像的形状从 (H, W) 转换为 (1, 1, H, W)
    binary_tensor = torch.tensor(binary_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

    return binary_tensor

def extract_and_resize(region_info, original_image):
    """
    从原图中提取区域，并调整为 128x128 的正方形图像。
    
    参数:
    region_info (list): 含有区域信息的字典列表。
    original_image (numpy.ndarray): 原始图像。
    
    """
    
    drops = { 'brightness': [], 'area': []}
    
    for region in region_info:
        
        label = region['label']
        mask = region['mask']
        center = region['center']
        
        # 1. 提取区域的边界框
        y_indices, x_indices = np.where(mask == 255)
        if len(y_indices) == 0 or len(x_indices) == 0:
            continue  # 如果没有区域，跳过

        # 计算边界框
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # 2. 创建一个黑色背景图像
        height = y_max - y_min + 1
        width = x_max - x_min + 1
        max_size = max(height, width)
        
        # 3. 创建正方形图像并设置中心
        square_image = np.zeros((max_size, max_size, 3), dtype=np.uint8)

        # 计算放置区域的起始坐标
        x_offset = (max_size - width) // 2
        y_offset = (max_size - height) // 2
        
        # 4. 将原区域放置到背景上
        #square_image[y_offset:y_offset + height, x_offset:x_offset + width] = original_image[y_min:y_max + 1, x_min:x_max + 1]
        square_image[y_offset:y_offset + height, x_offset:x_offset + width] = original_image[y_min:y_max + 1, x_min:x_max + 1] * (mask[y_min:y_max + 1, x_min:x_max + 1] // 255)[:, :, np.newaxis]

        # 5. 调整为 128x128
        resized_image = cv2.resize(square_image, (128, 128), interpolation=cv2.INTER_AREA)
        #d = Droplet(resized_image, region['brightness'], region['area'])

        drops['area'].append(region['area'])
        #drops['img'].append(resized_image)
        drops['brightness'].append(region['brightness'])
        #resized_images.append(resized_image)
    d = pd.DataFrame(drops)
    return d

import os
import uuid

def generate_unique_filename(directory, extension):
    # 生成一个随机的文件名
    while True:
        random_filename = f"{uuid.uuid4()}{extension}"
        full_path = os.path.join(directory, random_filename)
        
        # 检查文件名是否已存在
        if not os.path.exists(full_path):
            return full_path

def get_imglist(img,
                R=50,
                INCREASE = 20,     #开运算边界增强
                THRES_CIRCULARITY = 0.6,
                THRES_AREA = -1,
                BLUR_SIZE = -1
               ):
    HALF_SIZE = R//5
    KERNEL_SIZE = HALF_SIZE *2 +1
    if THRES_AREA == -1 :
        THRES_AREA = 3.14159 * R * R * 0.5
    if BLUR_SIZE == -1 :
        BLUR_SIZE = R//10
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用高斯模糊去噪声
    blurred_image = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
    thres = cv2.adaptiveThreshold(
        blurred_image,
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        6*R+1,  # blockSize
        2    # C，从计算出的阈值中减去的值，用于调节结果。
    )
    kernel = np.ones((INCREASE, INCREASE), np.uint8)
    th2 = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel)
    k1=np.zeros((KERNEL_SIZE,KERNEL_SIZE), np.uint8)
    cv2.circle(k1,(HALF_SIZE,HALF_SIZE),HALF_SIZE,(1,1,1),-1, cv2.LINE_AA)
    erode = cv2.morphologyEx(th2, cv2.MORPH_ERODE, k1)
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(erode, connectivity=8)
    
    # 创建一个空白图像以绘制膨胀后的区域
    dilated_image = np.zeros_like(th2)
    
    # 记录区域信息
    region_info = []
    dropletlist = []
    
    # 对每个连通区域进行膨胀
    for label in range(1, num_labels):  # 从 1 开始以跳过背景
        # 创建一个只包含当前区域的掩码
        mask = np.zeros_like(erode)
        mask[labels == label] = 255  # 只保留当前连通区域
    
        # 膨胀操作
        dilated_region = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k1)
    
        # 计算区域的面积
        area = cv2.countNonZero(dilated_region)
    
        # 计算轮廓并计算圆度
        contours, _ = cv2.findContours(dilated_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circularity = 0
        if contours:
            # 取第一个轮廓
            cnt = contours[0]
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                
        if circularity < THRES_CIRCULARITY or area < THRES_AREA :
            continue
        if GET_LIST == True:
            region_pixels = gray[mask > 0]  # 选择有效区域的像素
            
            if region_pixels.size > 0:
                average_brightness = np.mean(region_pixels)
            # 记录区域信息
            region_info.append({
                'label': label,
                'area': area,
                'brightness': average_brightness,
                'circularity': circularity,
                'mask': dilated_region,
                'center': centers[label]  # 记录中心位置
            })
        # 将膨胀后的区域合并到最终图像中
        #if SAVE_COMPARE == True:
        dilated_image[dilated_region > 0] = 255
    if GET_LIST == True:
        dropletlist = extract_and_resize(region_info, img)

    if SAVE_COMPARE == True:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img*5, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # 关闭坐标轴

        plt.subplot(1, 2, 2)
        plt.imshow(dilated_image, cmap= 'gray')
        plt.axis('off')
        fname = generate_unique_filename('./tmpshow', '.png')
        plt.savefig(fname)
        plt.close() 

    return dropletlist, dilated_image
    

def process_images_from_folder(folder_path):
    #all_images_list = []  # 用于存储所有图片的列表
    result = pd.DataFrame()
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):  # 检查图片扩展名
            # 生成完整的文件路径
            image_path = os.path.join(folder_path, filename)
            
            # 读取图片
            img = cv2.imread(image_path)
            if img is not None:
                # 对图片应用 get_imglist 函数
                img_list = get_imglist(img)
                # 合并到总列表中
                #all_images_list.extend(img_list)
                result = pd.concat([result, img_list], ignore_index=True) 
            else:
                print(f"Warning: Unable to read image {image_path}")

    return result


def get_single_from_folder(folder_path):
    cnt = 0
    result = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):  # 检查图片扩展名
            # 生成完整的文件路径
            image_path = os.path.join(folder_path, filename)
            
            # 读取图片
            img = cv2.imread(image_path)
            if img is not None:
                # 对图片应用 get_imglist 函数
                img_list = get_imglist(img)
                result.append(img_list)
                cnt = cnt + 1
                
            else:
                print(f"Warning: Unable to read image {image_path}")

    return result

def avgbright(df):
    total_area = df['area'].sum()
    return (df['brightness'] * df['area']).sum() / total_area



def expng(drops, filename):
    y = drops.brightness
    x = drops.area
    plt.figure() 
    plt.scatter(x, y, label=f'{filename}', s=1)
    plt.title(f'Scatter Plot for Drop {filename}')
    plt.xlabel('Area')
    plt.ylabel('Brightness')
    plt.xlim(3000, 26000)
    plt.legend()
    plt.grid(True)


    plt.savefig(f'{filename}.png')  # 保存为 PNG 文件
    plt.close() 



