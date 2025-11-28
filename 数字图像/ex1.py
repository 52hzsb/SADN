import numpy as np
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 定义一维高斯函数
def gaussian(x, mu, sigma):
    """
    计算一维高斯函数值
    公式：f(x) = (1/(σ√(2π))) * exp(-0.5*((x-μ)/σ)²)
    """
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))  # 归一化系数
    exponent = -0.5 * ((x - mu) / sigma) **2  # 指数部分
    return coefficient * np.exp(exponent)


x = np.linspace(-10, 10, 2000)  


parameters = [
    (0, 0.5, "σ=0.5（陡峭）"),   
    (0, 1, "σ=1（中等）"),      
    (0, 2, "σ=2（平缓）"),       
    (3, 1, "μ=3（右移）"),       
    (-2, 1, "μ=-2（左移）")     
]

# 创建画布
plt.figure(figsize=(12, 7))
plt.title("不同均值(μ)和标准差(σ)的一维高斯函数对比", fontsize=15)
plt.xlabel("x值", fontsize=12)
plt.ylabel("函数值 f(x)", fontsize=12)
plt.grid(linestyle="--", alpha=0.6)  # 网格线增强可读性


colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
line_styles = ["-", "--", "-.", ":", "-"]

for i, (mu, sigma, label) in enumerate(parameters):
    y = gaussian(x, mu, sigma)
    plt.plot(x, y, 
             color=colors[i], 
             linestyle=line_styles[i],
             linewidth=2.5, 
             label=f"μ={mu}, {label}")


plt.legend(fontsize=11, loc="upper right")
for mu, sigma, _ in parameters:
    plt.axvline(x=mu, color="gray", linestyle=":", alpha=0.3)  

# 调整布局并显示
plt.tight_layout()  
plt.show()