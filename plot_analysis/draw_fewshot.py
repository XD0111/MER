import matplotlib.pyplot as plt
import numpy as np

# 1. 数据准备
x_labels = ['0%', '20%', '40%', '60%', '80%', '100%']
f1_sub = [10.5, 36.1, 44.4, 44.3, 64.5, 65.2]
f1_cau = [17.0, 48.8, 51.8, 54.4, 55.3, 62.0]
f1_temp = [14.0, 43.1, 43.6, 44.5, 40.5, 50.2]
f1_cof = [24.6, 83.0, 83.7, 86.6, 87.1, 87.3]
f1_avg = [16.5, 52.8, 55.9, 57.5, 61.9, 66.2]

data_list = [f1_sub, f1_cau, f1_temp, f1_cof, f1_avg]
titles = ['Sub', 'Cau', 'Temp', 'Cof', 'Average']
colors = ['#5C80BC', '#E27D60', '#8AB17D', '#C06C84', '#2A9D8F']

# 2. 创建 2行3列 画布，加宽 figsize 以适应横向排版
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

# 3. 绘制前 5 个子图
for i in range(5):
    ax = axes[i]
    bars = ax.bar(x_labels, data_list[i], color=colors[i], width=0.55,
                  edgecolor='black', linewidth=0.8, alpha=0.85)

    ax.set_title(titles[i], fontsize=20, pad=18)
    ax.set_ylabel('F1 Score (%)', fontsize=16)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', labelsize=16)  # 将原来的 14 改大到 16，或者更大如 18
    ax.tick_params(axis='y', labelsize=16)
    # 样式美化
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 在柱子上添加数值标签
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 1.5,
                f'{yval}', ha='center', va='bottom', fontsize=16)

# 4. 设置第 6 个位置（右下角）作为图注区域
ax_legend = axes[5]
ax_legend.axis('off')

# 自定义图例句柄
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], ec="black", alpha=0.85) for i in range(5)]
# 【关键修改】：更大字体，更紧凑间距
ax_legend.legend(handles, titles, loc='center',
                 fontsize=20,              # [调大] 类别字体调大到 22
                 title_fontsize=22,        # [调大] 标题 "Event Types" 调大到 24
                 frameon=True, edgecolor='black', facecolor='white',
                 title="Event Types",
                 labelspacing=0.8,         # [紧凑] 调小行间距 (默认通常是 1.2 左右)
                 handlelength=1.5,         # [加大色块] 色块的宽度
                 handleheight=1.2,         # [加大色块] 色块的高度，配合大字体更好看
                 borderpad=0.8)            # [紧凑] 图例边框内壁与文字/色块的留白距离

# 5. 调整布局并保存
plt.tight_layout()
plt.savefig('study_fewshot.png', dpi=300, bbox_inches='tight')
plt.show()