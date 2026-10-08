import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# 1. 准备数据
x_vals = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
p_vals = [77.6, 76.4, 77.9, 78.1, 77.5, 77.7]
r_vals = [57.0, 58.4, 57.3, 57.6, 56.7, 56.8]
f1_vals = [65.5, 66.2, 65.8, 66.0, 65.2, 65.4]

# 2. 创建断轴图：上下两个子图，共享 x 轴
# 调整 ratios，腾出顶部空间
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, sharex=True, figsize=(7.8, 5.2),
    gridspec_kw={'height_ratios': [1, 1.4], 'hspace': 0.1}
)

# 3. 在两个坐标轴上都画同样的数据
for ax in [ax_top, ax_bottom]:
    ax.plot(x_vals, f1_vals, marker='D', linestyle='-', color='#2A9D8F',
            label='F1-score', linewidth=2.5, markersize=6)
    ax.plot(x_vals, p_vals, marker='o', linestyle='--', color='#5C80BC',
            label='Precision', linewidth=2, markersize=6)
    ax.plot(x_vals, r_vals, marker='s', linestyle='-.', color='#E27D60',
            label='Recall', linewidth=2, markersize=6)

# 4. 设置上下 y 轴范围及刻度
ax_top.set_ylim(75, 83)
ax_bottom.set_ylim(54, 69)
ax_top.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax_bottom.yaxis.set_major_locator(ticker.MultipleLocator(5))

# 5. 标注最大值
def mark_max(ax_list, x, y, color, marker, text_offset=(0.012, 0.4)):
    max_idx = np.argmax(y)
    max_x = x[max_idx]
    max_y = y[max_idx]
    for ax in ax_list:
        y_min, y_max = ax.get_ylim()
        if y_min <= max_y <= y_max:
            ax.scatter(max_x, max_y, color=color, s=80, marker=marker,
                       edgecolors='black', linewidths=0.8, zorder=10)
            ax.axhline(y=max_y, color=color, linestyle='--', linewidth=1.0, alpha=0.7)
            ax.text(max_x + text_offset[0], max_y + text_offset[1],
                    f'{max_y:.1f}', color=color, fontsize=16, fontweight='bold', zorder=11)

mark_max([ax_top, ax_bottom], x_vals, f1_vals, '#2A9D8F', 'D', text_offset=(0.012, 0.4))
mark_max([ax_top, ax_bottom], x_vals, p_vals,  '#5C80BC', 'o', text_offset=(0.012, 0.4))
mark_max([ax_top, ax_bottom], x_vals, r_vals,  '#E27D60', 's', text_offset=(0.012, 0.4))

# 6. 断轴样式设置
ax_top.spines['bottom'].set_visible(False)
ax_bottom.spines['top'].set_visible(False)
ax_top.tick_params(labelbottom=False, bottom=False)
ax_bottom.tick_params(top=False)

# 绘制断轴斜杠
d = 0.015
kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1.2)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax_bottom.transAxes)
ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# 7. 标签与刻度 (增大刻度数字，拉近 Score 标签)
ax_bottom.set_xlabel(r'Parameter $\epsilon$', fontsize=16)
fig.text(0.02, 0.5, 'Score (%)', va='center', rotation='vertical', fontsize=16)

ax_bottom.set_xticks(x_vals)
ax_bottom.tick_params(axis='x', labelsize=16)
ax_top.tick_params(axis='y', labelsize=16)
ax_bottom.tick_params(axis='y', labelsize=16)

# 8. 图注：改为单行 (ncol=3) 放在外部
ax_top.legend(
    loc='upper center',
    ncol=3,
    frameon=True,
    fontsize=14,
    edgecolor='black',
    facecolor='white',
    framealpha=1.0,
    handletextpad=0.5,
    columnspacing=1.0
)

# 9. 布局调整
plt.subplots_adjust(left=0.12, top=0.88, bottom=0.12)
# plt.savefig('parameter_epsilon_study_improved.pdf', dpi=300, bbox_inches='tight')
plt.savefig('study_epsilon.png', dpi=300, bbox_inches='tight')
plt.show()