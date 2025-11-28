import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
n = 20
corr_matrix = np.random.rand(n, n)
corr_matrix = (corr_matrix + corr_matrix.T) / 2
np.fill_diagonal(corr_matrix, 1.0)

plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(
    corr_matrix,
    cmap="RdYlGn",
    annot=False,
    linewidths=0.5,
    linecolor="gray"
)
cbar = plt.colorbar(heatmap.collections[0])
cbar.set_label("Correlation Score")
plt.title("Correlation Heatmap")
plt.show()