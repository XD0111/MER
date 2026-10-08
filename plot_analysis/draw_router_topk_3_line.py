import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# ====== 1. 准备数据 ======
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
categories = ['Sub', 'Cau', 'Temp', 'Cof', 'Average']
methods = ['Majority Voting', 'Weighted Voting', 'Router (top-k=3)']

f1_majority = [55.7, 59.4, 47.6, 87.6, 62.6]
f1_weighted = [55.9, 59.6, 49.7, 88.8, 63.5]
f1_full = [63.2, 61.2, 50.8, 86.9, 65.5]

data_matrix = np.array([f1_majority, f1_weighted, f1_full])
df = pd.DataFrame(data_matrix, columns=categories, index=methods)

# ====== 2. 绘制热力图 ======
fig, ax = plt.subplots(figsize=(9, 4), dpi=300)

# 使用 seaborn 绘制，cmap 选择蓝色渐变，颜色越深代表分数越高
sns.heatmap(df, annot=True, fmt='.1f', cmap='Blues', vmin=45, vmax=90,
            annot_kws={"size": 16, "weight": "bold"},
            cbar_kws={'label': 'F1-score (%)'}, ax=ax)

# ====== 3. 格式化图表 ======
ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, rotation=45, va='center', ha='right')

cbar = ax.collections[0].colorbar
cbar.ax.yaxis.label.set_size(14)

plt.tight_layout()
# plt.savefig('draw_router_topk_3.png', dpi=300, bbox_inches='tight')
plt.show()