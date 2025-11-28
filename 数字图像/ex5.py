import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def bilateral_filter_color_image(img_path, d=9, sigmaColor=75, sigmaSpace=75):
   
    # 读取真彩色图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误：无法读取图像 {img_path}")
        return
    
    # 转换为RGB格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 分离BGR通道
    b, g, r = cv2.split(img)
    
   
    b_filtered = cv2.bilateralFilter(b, d, sigmaColor, sigmaSpace)
    g_filtered = cv2.bilateralFilter(g, d, sigmaColor, sigmaSpace)
    r_filtered = cv2.bilateralFilter(r, d, sigmaColor, sigmaSpace)
    
    # 合并滤波后的通道
    img_filtered = cv2.merge([b_filtered, g_filtered, r_filtered])
    img_filtered_rgb = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB)
    
    # 展示结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("原图（真彩色）")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_filtered_rgb)
    plt.title(f"双边滤波后")
    plt.axis("off")
    
    plt.suptitle("真彩色图像双边滤波效果对比", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
 
    image_path = "fig.jpg"  
    bilateral_filter_color_image(image_path, d=9, sigmaColor=75, sigmaSpace=75)