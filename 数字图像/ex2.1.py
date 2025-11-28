import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


# 生成带高斯噪声的测试图像
def add_gaussian_noise(image, mean=0, var=0.001):
    """添加高斯噪声"""
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var**0.5, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return np.uint8(noisy_image * 255)


# 读取原始图像（替换为自己的图像路径）
original = cv2.imread("fig.jpg", cv2.IMREAD_GRAYSCALE)
if original is None:
    # 若图像不存在，生成测试图
    original = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(original, (50, 50), (200, 200), 128, -1)
    cv2.circle(original, (128, 128), 50, 200, -1)

# 生成带高斯噪声的图像
gaussian_noisy = add_gaussian_noise(original, var=0.005)  # 高斯噪声（方差0.005）


# 高斯滤波参数对比实验
# 实验1：同一模板尺寸(5x5)，不同σ
sigma_list = [0.5, 1.0, 2.0, 3.0]
gauss_same_size = [
    cv2.GaussianBlur(gaussian_noisy, (5, 5), sigmaX=sigma) 
    for sigma in sigma_list
]

# 实验2：同一σ(1.0)，不同模板尺寸
size_list = [(3, 3), (5, 5), (7, 7), (9, 9)]
gauss_same_sigma = [
    cv2.GaussianBlur(gaussian_noisy, size, sigmaX=1.0) 
    for size in size_list
]


# 可视化函数
def plot_images(images, titles, rows, cols, figsize=(15, 10)):
    """批量绘制图像"""
    plt.figure(figsize=figsize)
    for i in range(len(images)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# 实验1：同一尺寸(5x5)，不同σ的高斯滤波对比
plot_images(
    [gaussian_noisy] + gauss_same_size,
    ["原始高斯噪声图"] + [f"σ={s}" for s in sigma_list],
    1, 5, (16, 4)
)

# 实验2：同一σ(1.0)，不同尺寸的高斯滤波对比
plot_images(
    [gaussian_noisy] + gauss_same_sigma,
    ["原始高斯噪声图"] + [f"尺寸={s}" for s in size_list],
    1, 5, (16, 4)
)


# 结果分析
print("=== 同一模板尺寸(5x5)，不同σ的高斯滤波结果分析 ===")
print("1. σ越小，高斯核权重越集中在中心，滤波后图像越清晰，但去噪效果较弱；")
print("2. σ越大，高斯核权重分布越分散，去噪效果越强，但图像模糊程度越明显；")
print("3. 当σ超过2.0后，模糊加剧，细节损失严重，需根据实际需求平衡去噪与清晰度。")

print("\n=== 同一σ(1.0)，不同模板尺寸的高斯滤波结果分析 ===")
print("1. 模板尺寸越小(3x3)，参与滤波的邻域像素少，去噪不充分，图像仍有明显噪声；")
print("2. 模板尺寸增大(5x5/7x7)，去噪效果提升，图像更平滑，但边缘细节逐渐模糊；")
print("3. 模板尺寸过大(9x9)，去噪效果提升有限，却导致过度模糊，性价比降低；")
print("4. 建议模板尺寸与σ匹配（通常取尺寸=6σ+1），确保滤波效果与效率平衡。")