import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 全局统一字体配置 --------------------------
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['font.size'] = 12

# 1. 生成带噪声的测试图像
def add_gaussian_noise(image, mean=0, var=0.001):
    """添加高斯噪声"""
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var**0.5, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return np.uint8(noisy_image * 255)

def add_salt_pepper_noise(image, prob=0.05):
    """添加椒盐噪声"""
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

# 读取原始图像
original = cv2.imread("fig.jpg", cv2.IMREAD_GRAYSCALE)
if original is None:
    original = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(original, (50, 50), (200, 200), 128, -1)
    cv2.circle(original, (128, 128), 50, 200, -1)

# 生成带噪声的图像
gaussian_noisy = add_gaussian_noise(original, var=0.005)
salt_pepper_noisy = add_salt_pepper_noise(original, prob=0.08)

# 2. 滤波处理
mean_kernels = [(3, 3), (5, 5), (7, 7), (9, 9)]

# 高斯噪声图像的滤波结果
mean_gaussian = [cv2.blur(gaussian_noisy, k) for k in mean_kernels]
gauss_gaussian = cv2.GaussianBlur(gaussian_noisy, (5, 5), sigmaX=1.0)

# 椒盐噪声图像的滤波结果
mean_saltpepper = [cv2.blur(salt_pepper_noisy, k) for k in mean_kernels]
gauss_saltpepper = cv2.GaussianBlur(salt_pepper_noisy, (5, 5), sigmaX=1.0)

# 3. 统一参数的可视化函数
def plot_comparison(images, titles, figsize=(18, 12), nrows=2, ncols=3):
    """统一格式的对比图绘制函数"""
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    # 将axes转换为1D数组以便遍历
    axes = axes.flatten()
    
    for i in range(len(images)):
        axes[i].imshow(images[i], cmap="gray")
        # 强制固定标题字体大小和样式
        axes[i].set_title(titles[i], fontsize=16, fontweight='normal', pad=8)
        axes[i].axis("off")
    
    # 隐藏多余的子图
    for i in range(len(images), len(axes)):
        axes[i].axis("off")
    
    # -------------------------- 关键修改：使用tight_layout统一控制布局 --------------------------
    plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)  # 统一控制所有间距
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, 
                       hspace=0.15, wspace=0.01)  # 统一调整边距和间距
    
    plt.show()

# -------------------------- 关键修改：使用类或全局配置确保参数一致性 --------------------------
class PlotConfig:
    """统一绘图配置类"""
    def __init__(self):
        self.figsize = (18, 12)
        self.nrows = 2
        self.ncols = 3
        self.fontsize = 16
        self.hspace = 0.15
        self.wspace = 0.001

# 创建全局配置实例
config = PlotConfig()

# 4.1 高斯噪声下：均值滤波 vs 高斯滤波
plot_comparison(
    images=[gaussian_noisy, gauss_gaussian] + mean_gaussian,
    titles=["原始高斯噪声图", "高斯滤波(5x5,σ=1)"] + [f"均值滤波{k}" for k in mean_kernels],
    figsize=config.figsize,
    nrows=config.nrows,
    ncols=config.ncols
)

# 4.2 椒盐噪声下：均值滤波 vs 高斯滤波
plot_comparison(
    images=[salt_pepper_noisy, gauss_saltpepper] + mean_saltpepper,
    titles=["原始椒盐噪声图", "高斯滤波(5x5,σ=1)"] + [f"均值滤波{k}" for k in mean_kernels],
    figsize=config.figsize,
    nrows=config.nrows,
    ncols=config.ncols
)