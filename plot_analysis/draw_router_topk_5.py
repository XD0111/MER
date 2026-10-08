import matplotlib.pyplot as plt
import numpy as np

# ====== 1. 准备数据 ======
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
categories = ['Sub', 'Cau', 'Temp', 'Cof', 'Average']

# 对应类别的 F1-score 数据
f1_majority = [51.6, 58.8, 45.1, 85.7, 58.4]
f1_weighted = [53.6, 58.2, 51.2, 85.8, 60.2]
f1_full = [64.7, 61.3, 49.2, 87.3, 65.6]

# ====== 2. 设置画布与柱状图参数 ======
x = np.arange(len(categories))
width = 0.3

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# color1 = '#D8C6FF'   # 淡紫
# color2 = '#FFF0A6'   # 浅鹅黄
# color3 = '#FFB3A7'   # 嫩草绿
color1 = '#CAB8FF'
color2 = '#FFE082'
color3 = '#FFB5A7'

rects1 = ax.bar(x - width, f1_majority, width, label='Majority Voting', color=color1)
rects2 = ax.bar(x, f1_weighted, width, label='Weighted Voting', color=color2)
rects3 = ax.bar(x + width, f1_full, width, label='Router (top-K=5)', color=color3)

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
plt.savefig('router_topk_5.png', dpi=300, bbox_inches='tight')
plt.show()