import matplotlib.pyplot as plt
import numpy as np

# ====== 1. 准备数据 ======
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
categories = ['Sub', 'Cau', 'Temp', 'Cof', 'Average']

# 对应类别的 F1-score 数据
f1_majority = [55.7, 59.4, 47.6, 87.6, 62.6]
f1_weighted = [55.9, 59.6, 49.7, 88.8, 63.5]
f1_full = [63.2, 61.2, 50.8, 86.9, 65.5]

# ====== 2. 设置画布与柱状图参数 ======
x = np.arange(len(categories))
width = 0.3

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

color1 = '#CAB8FF'
color2 = '#FFE082'
color3 = '#FFB5A7'

rects1 = ax.bar(x - width, f1_majority, width, label='Majority Voting', color=color1)
rects2 = ax.bar(x, f1_weighted, width, label='Weighted Voting', color=color2)
rects3 = ax.bar(x + width, f1_full, width, label='Router (top-k=3)', color=color3)

# ====== 3. 格式化图表 ======
# x轴、y轴标题字号
ax.set_ylabel('F1-score (%)', fontsize=20)

# x轴刻度标签字号
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=18)

# y轴刻度标签字号
ax.tick_params(axis='y', labelsize=18)

ax.set_ylim(40, 100)

ax.legend(loc='upper left', fontsize=16)

# ====== 4. 添加数据标签 ======
ax.bar_label(rects1, padding=4, fmt='%.1f', fontsize=14, fontweight='bold')
ax.bar_label(rects2, padding=4, fmt='%.1f', fontsize=14, fontweight='bold')
ax.bar_label(rects3, padding=4, fmt='%.1f', fontsize=14, fontweight='bold')

# ====== 5. 渲染与保存 ======
plt.tight_layout()
plt.savefig('router_topk_3.png', dpi=300, bbox_inches='tight')
plt.show()