import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 实验一：双边滤波器及三种滤波器综合分析（参数优化） --------------------------
def experiment_bilateral_filter(img_path):
    # 1. 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误：无法读取图像 {img_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 图像填充
    kernel_size = 7  # 增大核大小，增强平滑对比
    pad_size = kernel_size // 2
    img_padded = cv2.copyMakeBorder(
        img_gray,
        top=pad_size, bottom=pad_size, left=pad_size, right=pad_size,
        borderType=cv2.BORDER_REPLICATE
    )

    # 3. 双边滤波（增大sigma，增强边缘保留特性）
    bilateral_filtered = cv2.bilateralFilter(img_padded, d=kernel_size, sigmaColor=100, sigmaSpace=100)
    bilateral_filtered = bilateral_filtered[pad_size:-pad_size, pad_size:-pad_size]

    # 4. 均值滤波（大核增强模糊效果）
    mean_filtered = cv2.blur(img_gray, ksize=(kernel_size, kernel_size))

    # 5. 高斯滤波（调整sigma，增强平滑与边缘过渡）
    gaussian_filtered = cv2.GaussianBlur(img_gray, ksize=(kernel_size, kernel_size), sigmaX=2.5)

    # 6. 结果展示
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title("原图（RGB）")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(mean_filtered, cmap="gray")
    plt.title(f"均值滤波（{kernel_size}×{kernel_size}）")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(gaussian_filtered, cmap="gray")
    plt.title(f"高斯滤波（{kernel_size}×{kernel_size}，σ=2.5）")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(bilateral_filtered, cmap="gray")
    plt.title(f"双边滤波（d={kernel_size}, σ_color=100, σ_space=100）")
    plt.axis("off")

    plt.suptitle("实验一：均值、高斯、双边滤波器对比（参数优化）", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.show()

# -------------------------- 实验二：自定义均值滤波器（参数优化） --------------------------
def experiment_custom_mean_filter(img_path):
    # 1. 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误：无法读取图像 {img_path}")
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. 自定义均值滤波器函数
    def custom_mean_filter(img, kernel_size):
        h, w = img.shape
        pad_size = kernel_size // 2
        img_padded = cv2.copyMakeBorder(
            img,
            top=pad_size, bottom=pad_size, left=pad_size, right=pad_size,
            borderType=cv2.BORDER_REPLICATE
        )
        output = np.zeros_like(img, dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                window = img_padded[i:i+kernel_size, j:j+kernel_size]
                output[i, j] = np.mean(window).astype(np.uint8)
        return output

    # 3. 应用自定义均值滤波器（5×5 核，增强模糊对比）
    kernel_size = 5
    custom_mean_filtered = custom_mean_filter(img_gray, kernel_size)

    # 4. 与 OpenCV 内置均值滤波对比
    opencv_mean_filtered = cv2.blur(img_gray, ksize=(kernel_size, kernel_size))

    # 5. 结果展示
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("原图（RGB）")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(custom_mean_filtered, cmap="gray")
    plt.title(f"自定义均值滤波（{kernel_size}×{kernel_size}）")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(opencv_mean_filtered, cmap="gray")
    plt.title(f"OpenCV 均值滤波（{kernel_size}×{kernel_size}）")
    plt.axis("off")

    plt.suptitle("实验二：自定义均值滤波器与 OpenCV 均值滤波对比（参数优化）", fontsize=16, y=0.85)
    plt.tight_layout()
    plt.show()

# -------------------------- 实验三：锐化滤波器（参数优化，增强效果差异） --------------------------
def experiment_sharpening_filters(img_path):
    # 1. 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误：无法读取图像 {img_path}")
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. 定义锐化滤波器核
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    robert_1 = np.array([[1, 0], [0, -1]], dtype=np.float32)
    robert_2 = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    # 3. 锐化滤波函数（调整边缘权重，增强锐化效果）
    def sharpen_filter(img, kernel, weight=0.8):
        edge = cv2.filter2D(img.astype(np.float32), -1, kernel)
        edge = cv2.normalize(edge, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        sharpened = cv2.addWeighted(img, 1.0, edge, weight, 0)
        return sharpened

    # 4. 分别应用四种锐化滤波（调整权重，突出各方法特色）
    sobel_x_sharp = sharpen_filter(img_gray, sobel_x, 0.8)
    sobel_y_sharp = sharpen_filter(img_gray, sobel_y, 0.8)
    sobel_combined = cv2.addWeighted(sobel_x_sharp, 0.5, sobel_y_sharp, 0.5, 0)

    prewitt_x_sharp = sharpen_filter(img_gray, prewitt_x, 0.8)
    prewitt_y_sharp = sharpen_filter(img_gray, prewitt_y, 0.8)
    prewitt_combined = cv2.addWeighted(prewitt_x_sharp, 0.5, prewitt_y_sharp, 0.5, 0)

    robert_1_sharp = sharpen_filter(img_gray, robert_1, 0.8)
    robert_2_sharp = sharpen_filter(img_gray, robert_2, 0.8)
    robert_combined = cv2.addWeighted(robert_1_sharp, 0.5, robert_2_sharp, 0.5, 0)

    laplacian_sharp = sharpen_filter(img_gray, laplacian, 1.0)  # 拉普拉斯权重加大，突出细节增强

    # 5. 结果展示
    plt.figure(figsize=(18, 12))

    plt.subplot(3, 3, 1)
    plt.imshow(img_rgb)
    plt.title("原图（RGB）")
    plt.axis("off")

    plt.subplot(3, 3, 2)
    plt.imshow(sobel_x_sharp, cmap="gray")
    plt.title("Sobel 水平锐化（强边缘）")
    plt.axis("off")
    plt.subplot(3, 3, 3)
    plt.imshow(sobel_y_sharp, cmap="gray")
    plt.title("Sobel 垂直锐化（强边缘）")
    plt.axis("off")
    plt.subplot(3, 3, 4)
    plt.imshow(sobel_combined, cmap="gray")
    plt.title("Sobel 组合锐化（强边缘）")
    plt.axis("off")

    plt.subplot(3, 3, 5)
    plt.imshow(prewitt_x_sharp, cmap="gray")
    plt.title("Prewitt 水平锐化（宽边缘）")
    plt.axis("off")
    plt.subplot(3, 3, 6)
    plt.imshow(prewitt_y_sharp, cmap="gray")
    plt.title("Prewitt 垂直锐化（宽边缘）")
    plt.axis("off")
    plt.subplot(3, 3, 7)
    plt.imshow(prewitt_combined, cmap="gray")
    plt.title("Prewitt 组合锐化（宽边缘）")
    plt.axis("off")

    plt.subplot(3, 3, 8)
    plt.imshow(robert_combined, cmap="gray")
    plt.title("Robert 组合锐化（细边缘）")
    plt.axis("off")

    plt.subplot(3, 3, 9)
    plt.imshow(laplacian_sharp, cmap="gray")
    plt.title("拉普拉斯锐化（细节增强）")
    plt.axis("off")

    plt.suptitle("实验三：四种锐化滤波器对比（参数优化）", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.show()

# -------------------------- 主函数：运行所有实验 --------------------------
if __name__ == "__main__":
    image_path = "fig.jpg"  # 替换为你的图像路径
    
    print("正在运行实验一：双边滤波器及三种滤波器综合分析...")
    experiment_bilateral_filter(image_path)
    
    print("正在运行实验二：自定义均值滤波器...")
    experiment_custom_mean_filter(image_path)
    
    print("正在运行实验三：锐化滤波器...")
    experiment_sharpening_filters(image_path)

    print("所有实验运行完毕！")