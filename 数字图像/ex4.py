import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def improved_mean_filter_strong(img_gray, kernel_size=7, edge_weight=0.2, edge_threshold=30):
   
    h, w = img_gray.shape
    pad_size = kernel_size // 2
    img_padded = cv2.copyMakeBorder(
        img_gray, 
        pad_size, pad_size, pad_size, pad_size, 
        cv2.BORDER_REPLICATE
    )
    output = np.zeros_like(img_gray, dtype=np.uint8)
    
    # 边缘检测（Sobel梯度计算，ksize=5增强梯度敏感性）
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
    edge = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_mask = edge > edge_threshold  # 严格判定边缘
    
    for i in range(h):
        for j in range(w):
            window = img_padded[i:i+kernel_size, j:j+kernel_size]
            if edge_mask[i, j]:
                # 边缘区域：仅用edge_weight比例的邻域均值，保留(1-edge_weight)的原始像素
                output[i, j] = np.uint8(edge_weight * np.mean(window) + (1 - edge_weight) * img_gray[i, j])
            else:
                # 非边缘区域：正常均值滤波
                output[i, j] = np.uint8(np.mean(window))
    return output


def verify_improved_mean_filter_strong(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("图像读取失败！")
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 原始均值滤波（7×7，模糊效果极强）
    mean_filtered = cv2.blur(img_gray, (7, 7))
    # 改进均值滤波（7×7，对比极明显）
    improved_filtered = improved_mean_filter_strong(
        img_gray, 
        kernel_size=7, 
        edge_weight=0.2, 
        edge_threshold=30
    )
    
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("原图（RGB）")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(mean_filtered, cmap="gray")
    plt.title("原始均值滤波（7×7）")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(improved_filtered, cmap="gray")
    plt.title("改进均值滤波（7×7）")
    plt.axis("off")
    
    plt.suptitle("改进均值滤波（增强版）vs 原始均值滤波（效果对比强化）", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    # 局部放大对比（聚焦发丝和猫咪边缘）
    plt.figure(figsize=(12, 4))
    # 原始均值滤波局部
    plt.subplot(1, 2, 1)
    plt.imshow(mean_filtered[50:200, 50:200], cmap="gray")
    plt.title("原始均值滤波（局部：发丝+猫咪边缘）")
    plt.axis("off")
    # 改进均值滤波局部
    plt.subplot(1, 2, 2)
    plt.imshow(improved_filtered[50:200, 50:200], cmap="gray")
    plt.title("改进均值滤波（局部：发丝+猫咪边缘）")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# 运行验证
if __name__ == "__main__":
    img_path = "fig.jpg"  # 替换为你的图像路径
    verify_improved_mean_filter_strong(img_path)