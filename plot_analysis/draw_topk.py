import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
top_k = [1, 2, 3, 4, 5]
precision = [70.7, 76.4, 78.1, 77.8, 77.7]
recall = [33.6, 58.4, 56.8, 56.6, 57.1]
f1_score = [43.4, 66.2, 65.5, 65.2, 65.6]

x = np.arange(len(top_k))
width = 0.3

# 2. 创建画布
fig, ax1 = plt.subplots(figsize=(8, 5))

# 【核心修改 1】：替换为原图精准提取的 Hex 颜色
precision_color = '#6FB9E9'  # 明亮的天蓝色
recall_color = '#FF9EB5'     # 清新的茱萸粉
f1_color = '#FBC02D'         # 亮眼的向日葵黄

# --- 绘制左侧 Y 轴数据 (Precision & Recall) ---
rects1 = ax1.bar(
    x - width/2, precision, width,
    label='Precision', color=precision_color, edgecolor='white'
)
rects2 = ax1.bar(
    x + width/2, recall, width,
    label='Recall', color=recall_color, edgecolor='white'
)

# 【核心修改 2】：左侧 Y 轴范围严格对齐原图 (30 到 85)
ax1.set_ylim(30, 85)
ax1.set_xticks(x)
ax1.set_xticklabels(top_k)

# --- 绘制右侧 Y 轴数据 (F1-score) ---
ax2 = ax1.twinx()
line = ax2.plot(
    x, f1_score,
    label='F1-score',
    color=f1_color,
    marker='o',
    markersize=5,
    linewidth=1.5
)

# 【核心修改 3】：右侧 Y 轴范围严格对齐原图 (40 到 80)
ax2.set_yticks(np.arange(40, 81, 10))

# 3. 添加辅助虚线
ax1.axhline(y=78.1, color=precision_color, linestyle='--', linewidth=1, alpha=0.5)
ax1.axhline(y=58.4, color=recall_color, linestyle='--', linewidth=1, alpha=0.5)
ax2.axhline(y=66.2, color=f1_color, linestyle='--', linewidth=1, alpha=0.5)

# 4. 添加数值标注
# 【核心修改 4】：精准微调文本坐标，使其与对应的柱子或折线点完美对齐
ax1.text(1.7, 78.1 + 0.5, '78.1', color=precision_color, fontsize=12, fontweight='bold')
ax1.text(1 , 58.4 + 0.5, '58.4', color=recall_color, fontsize=12, fontweight='bold')
ax2.text(1 , 66.2 + 0.5, '66.2', color=f1_color, fontsize=12, fontweight='bold')

# 5. 图例合并与轴标签字体大小 (你原先设置的标签大小)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12)
ax1.set_xlabel('Value of Top-K', fontsize=16)
ax1.set_ylabel('Precision & Recall (%)', fontsize=16)
ax2.set_ylabel('F1-score (%)', fontsize=16)

# ==========================================
# 【新增】：调大 X 轴和两侧 Y 轴的刻度(数字)字体大小
# ==========================================
ax1.tick_params(axis='both', which='major', labelsize=16)
ax2.tick_params(axis='y', which='major', labelsize=16)

# 6. 细节修饰
plt.tight_layout()

# 7. 保存图片 (使用 bbox_inches='tight' 确保变大后的字体不会被裁掉)
plt.savefig('topk.png', dpi=300, bbox_inches='tight')

plt.show()