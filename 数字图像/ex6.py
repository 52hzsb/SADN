import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'正在使用设备: {device}')

# ===================== 1. DnCNN 模型核心代码（带详细注释） =====================
class DnCNN(nn.Module):
    def __init__(self, num_layers=17, num_features=64, in_channels=1, out_channels=1):
        """
        DnCNN 模型初始化
        :param num_layers: 网络总层数
        :param num_features: 中间卷积层的输出特征图数量
        :param in_channels: 输入图像通道数（灰度图为1，RGB图为3）
        :param out_channels: 输出图像通道数（与输入相同）
        """
        super(DnCNN, self).__init__()
        
        # 构建网络层列表
        layers = []
        
        # 第一层：卷积 + ReLU。这一层没有 Batch Normalization
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # 中间层：共 (num_layers - 2) 层。每一层都是 Conv + BN + ReLU
        # Batch Normalization (BN) 是 DnCNN 论文中提出的关键改进之一，它能加速训练并防止过拟合
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features)) # BN层
            layers.append(nn.ReLU(inplace=True))
        
        # 最后一层：卷积层。输出通道数等于输入通道数，用于预测噪声残差
        # 这一层没有激活函数，因为残差可以是正或负的
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=False))
        
        # 使用 Sequential 容器来按顺序执行所有层
        self.dncnn = nn.Sequential(*layers)
        
        # 初始化网络权重
        self._initialize_weights()

    def _initialize_weights(self):
        """
        自定义权重初始化方法
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层权重使用 Kaiming 正态分布初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BN层权重初始化为1，偏置初始化为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播
        :param x: 输入的带噪声图像张量
        :return: 去噪后的图像张量
        """
        # 模型预测噪声残差
        residual = self.dncnn(x)
        # 从输入图像中减去预测的噪声，得到干净的图像
        return x - residual

# ===================== 2. 数据集定义 =====================
class DenoisingDataset(Dataset):
    def __init__(self, image_dir, patch_size=50, sigma=25):
        """
        用于图像去噪的数据集类
        :param image_dir: 干净图像所在的目录
        :param patch_size: 随机裁剪的图像块大小
        :param sigma: 要添加的高斯噪声的标准差
        """
        self.image_paths = sorted(list(Path(image_dir).glob('*.png'))) # 假设图像都是png格式
        self.patch_size = patch_size
        self.sigma = sigma

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像并转换为灰度图
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        
        # 随机裁剪一个图像块（数据增强）
        h, w = img.shape
        top = np.random.randint(0, h - self.patch_size)
        left = np.random.randint(0, w - self.patch_size)
        patch = img[top:top+self.patch_size, left:left+self.patch_size]

        # 数据预处理：归一化到 [0, 1] 范围，并转换为 float32
        patch = patch.astype(np.float32) / 255.0
        
        # 添加高斯噪声
        noise = np.random.normal(0, self.sigma / 255.0, patch.shape).astype(np.float32)
        noisy_patch = patch + noise
        
        # 转换为 PyTorch 张量，并增加一个通道维度 (H, W) -> (C, H, W)
        clean_tensor = torch.from_numpy(patch).unsqueeze(0)
        noisy_tensor = torch.from_numpy(noisy_patch).unsqueeze(0)
        
        return noisy_tensor, clean_tensor

# ===================== 3. 训练函数 =====================
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    """
    训练模型
    """
    model.train() # 设置模型为训练模式
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (noisy_imgs, clean_imgs) in enumerate(train_loader):
            # 将数据移到指定设备
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(noisy_imgs)
            
            # 计算损失
            loss = criterion(outputs, clean_imgs)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 打印每轮的平均损失
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}')
    
    print('训练完成!')
    return model

# ===================== 4. 测试/验证函数 =====================
def test_model(model, test_image_path, sigma=25):
    """
    测试模型
    """
    model.eval() # 设置模型为评估模式
    
    # 读取测试图像
    img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"错误：无法读取测试图像 {test_image_path}")
        return

    # 预处理
    img_clean = img.astype(np.float32) / 255.0
    
    # 添加噪声
    noise = np.random.normal(0, sigma / 255.0, img_clean.shape).astype(np.float32)
    img_noisy = img_clean + noise
    
    # 转换为张量
    img_noisy_tensor = torch.from_numpy(img_noisy).unsqueeze(0).unsqueeze(0).to(device)
    
    # 在无梯度的上下文中进行预测，以节省计算资源
    with torch.no_grad():
        img_denoised_tensor = model(img_noisy_tensor)
    
    # 将张量转换回图像格式
    img_denoised = img_denoised_tensor.squeeze(0).squeeze(0).cpu().numpy()
    
    # 确保像素值在 [0, 1] 范围内
    img_denoised = np.clip(img_denoised, 0, 1)
    
    # 转换为 uint8 用于显示
    img_clean_display = (img_clean * 255).astype(np.uint8)
    img_noisy_display = (img_noisy * 255).astype(np.uint8)
    img_denoised_display = (img_denoised * 255).astype(np.uint8)
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_clean_display, cmap='gray')
    plt.title('原图 (干净)')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img_noisy_display, cmap='gray')
    plt.title(f'带噪图像 (σ={sigma})')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_denoised_display, cmap='gray')
    plt.title('DnCNN 去噪结果')
    plt.axis('off')
    
    plt.suptitle('DnCNN 去噪效果对比')
    plt.show()

# ===================== 5. 主函数：执行训练和测试 =====================
if __name__ == '__main__':
    # --- 实验参数设置 ---
    TRAIN_IMAGE_DIR = 'path/to/your/train/images' # 替换为你的训练集图片目录
    TEST_IMAGE_PATH = 'path/to/your/test/image.png' # 替换为你的测试图片路径
    MODEL_SAVE_PATH = 'dncnn_model.pth'
    
    PATCH_SIZE = 50
    SIGMA = 25  # 训练和测试时使用的噪声强度
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # --- 1. 准备数据集 ---
    print("准备数据集...")
    try:
        train_dataset = DenoisingDataset(TRAIN_IMAGE_DIR, PATCH_SIZE, SIGMA)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"数据集准备完成，包含 {len(train_dataset)} 张图像。")
    except Exception as e:
        print(f"数据集准备失败: {e}")
        print("提示：请确保 TRAIN_IMAGE_DIR 路径正确，并且目录下有图片文件。")
        exit()

    # --- 2. 初始化模型、损失函数和优化器 ---
    print("初始化模型...")
    model = DnCNN(num_layers=17, num_features=64, in_channels=1, out_channels=1).to(device)
    criterion = nn.MSELoss() # 使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 3. 训练模型 ---
    print("开始训练...")
    trained_model = train_model(model, train_loader, criterion, optimizer, EPOCHS)
    
    # --- 4. 保存模型 ---
    torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
    print(f"模型已保存至 {MODEL_SAVE_PATH}")
    
    # --- 5. 测试模型 ---
    print("开始测试...")
    # 加载保存的模型
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to(device)
    test_model(model, TEST_IMAGE_PATH, SIGMA)