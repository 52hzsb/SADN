import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 数据预处理 (Data Preprocessing)
# ==========================================
def load_data(filename='new_mnist.json'):
    print(f"正在加载数据集: {filename} ...")
    try:
        with open(filename, 'r') as f:
            # 根据 Source 47，数据集结构为 train_set, dev_set, test_set
            train_set, dev_set, test_set = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filename}。请确保文件在当前目录下。")
        return None, None, None

    # 定义数据转换函数：将 list 转换为 Tensor 并 reshape
    # Source 40: 图片是长度为 784 的向量，需要 reshape 为 [28, 28]
    def process_set(dataset):
        images = dataset[0] # 图片数据 [N, 784]
        labels = dataset[1] # 标签数据 [N, 1] 或 [N]
        
        # 转换为 Tensor
        images_tensor = torch.tensor(images, dtype=torch.float32)
        # 根据 Source 31, 图像大小为 28x28。PyTorch CNN 需要 [Batch, Channel, Height, Width]
        images_tensor = images_tensor.view(-1, 1, 28, 28) 
        
        # 标签转换为 LongTensor (CrossEntropyLoss 需要)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # 如果标签是 [[1], [2]] 这种形状，需要压扁为 [1, 2]
        if labels_tensor.dim() > 1:
            labels_tensor = labels_tensor.squeeze()
            
        return TensorDataset(images_tensor, labels_tensor)

    train_data = process_set(train_set)
    dev_data = process_set(dev_set)
    test_data = process_set(test_set)
    
    # 打印数据集大小
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(dev_data)}")
    print(f"测试集样本数: {len(test_data)}")
    
    return train_data, dev_data, test_data

# ==========================================
# 2. 模型构建 (Model Construction)
# ==========================================
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # C1 卷积层: 输入 1通道, 输出 6通道, 核大小 5x5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # S2 汇聚层: MaxPool, 核大小 2x2, 步长 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # C3 卷积层: 输入 6通道, 输出 16通道, 核大小 5x5
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # S4 汇聚层: MaxPool, 核大小 2x2, 步长 2
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层输入计算: 
        # 输入 28x28 -> Conv1(5x5) -> 24x24 -> Pool2(2x2) -> 12x12
        # 12x12 -> Conv3(5x5) -> 8x8 -> Pool4(2x2) -> 4x4
        # 最终特征图大小: 16通道 * 4 * 4
        self.fc5 = nn.Linear(16 * 4 * 4, 120)
        self.fc6 = nn.Linear(120, 84)
        self.fc7 = nn.Linear(84, 10) # 输出 10 个类别

    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = self.pool2(F.relu(self.conv1(x)))
        x = self.pool4(F.relu(self.conv3(x)))
        
        # 展平 (Flatten) 用于全连接层
        x = x.view(-1, 16 * 4 * 4)
        
        # 全连接层
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x) # 最后一层通常不加激活函数，CrossEntropyLoss 会处理 softmax
        return x

# ==========================================
# 3. 训练与评价 (Training & Evaluation)
# ==========================================
def train_model(model, train_loader, dev_loader, epochs=10, learning_rate=0.01):
    # 损失函数: 交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 优化器: 梯度下降法 (SGD)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    train_losses = []
    dev_accuracies = []

    print(f"\n开始训练，共 {epochs} 个 Epoch...")
    
    for epoch in range(epochs):
        model.train() # 设置为训练模式
        running_loss = 0.0
        
        for images, labels in train_loader:
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(images)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 每个 Epoch 结束后在验证集上评估准确率
        acc = evaluate_model(model, dev_loader)
        dev_accuracies.append(acc)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Dev Accuracy: {acc*100:.2f}%")
        
    return train_losses, dev_accuracies

def evaluate_model(model, data_loader):
    model.eval() # 设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad(): # 不计算梯度
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# ==========================================
# 4. 主程序 (Main Runner)
# ==========================================
if __name__ == '__main__':
    # 设置随机种子保证可复现性
    torch.manual_seed(42)

    # 1. 加载数据
    train_data, dev_data, test_data = load_data('new_mnist.json')
    
    if train_data:
        # 创建 DataLoader
        batch_size = 64
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # 2. 实例化模型
        model = LeNet5()
        print("\n模型结构:")
        print(model)

        # 3. 训练模型
        # 实验要求：根据算力条件设计合理的训练次数
        # 这里设置为 20 次 epoch，对于 MNIST 1000条样本通常足够收敛
        epochs = 20 
        lr = 0.01
        losses, accuracies = train_model(model, train_loader, dev_loader, epochs, lr)

        # 4. 最终测试集评价
        test_acc = evaluate_model(model, test_loader)
        print(f"\n训练结束. 测试集准确率 (Test Accuracy): {test_acc*100:.2f}%")

        # 5. 可视化训练过程
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(accuracies, label='Dev Accuracy', color='orange')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 6. 模型预测示例 (Model Prediction)
        # 获取一张测试图片进行展示
        data_iter = iter(test_loader)
        images, labels = next(data_iter)
        
        idx = 0
        single_img = images[idx]
        true_label = labels[idx].item()
        
        # 预测
        model.eval()
        with torch.no_grad():
            output = model(single_img.unsqueeze(0)) # 增加 batch 维度
            pred_label = output.argmax(dim=1).item()
            
        print(f"\n预测示例:")
        print(f"真实标签: {true_label}")
        print(f"模型预测: {pred_label}")
        
        # 显示图片 (参考 Source 66-69 的逻辑)
        plt.imshow(single_img.squeeze(), cmap='gray')
        plt.title(f"True: {true_label}, Pred: {pred_label}")
        plt.show()