import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ['Mention', 'Sentence', 'Graph', 'Context', 'Path']
sizes = [13.16 / 3, 21.63 / 3, 100 / 3, 100 / 3, 65.52 / 3]

colors = ['#D7F9E8', '#E7D6FC', '#E0E0E0', '#74D3E9', '#FFB97F']

# 更适合论文排版的横向长方形画布
fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=300)

# 圆环图：缩小一点，但不要太小
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=None,
    colors=colors,
    autopct='%1.1f%%',
    startangle=140,
    pctdistance=0.76,
    radius=0.88,   # 圆整体略缩小
    wedgeprops={
        'edgecolor': 'white',
        'linewidth': 1.0,
        'width': 0.32
    },
    textprops={
        'fontsize': 10,
        'fontweight': 'bold',
        'color': '#3a3a3a',
        'family': 'DejaVu Sans'
    }
)

total = sum(sizes)
percentages = [s / total * 100 for s in sizes]

small_idx = 0
# 小扇区文字上下稍微错开，但不要太夸张
small_y_offsets = [0.08, 0.00, -0.08]

for wedge, autotext, pct in zip(wedges, autotexts, percentages):
    angle = (wedge.theta1 + wedge.theta2) / 2
    angle_rad = np.deg2rad(angle)

    if pct >= 10:
        # 大扇区数字放在圆环中部
        r = 0.72
        x = r * np.cos(angle_rad)
        y = r * np.sin(angle_rad)
        autotext.set_position((x, y))
        autotext.set_ha('center')
        autotext.set_va('center')
        autotext.set_fontsize(10)
    else:
        text_str = autotext.get_text()
        autotext.set_text('')

        # 引导线起点：外圆边缘
        r_start = 0.88
        x_start = r_start * np.cos(angle_rad)
        y_start = r_start * np.sin(angle_rad)

        # 关键：把小扇区数字拉近，不要离太远
        x_end = 1.02 * np.sign(np.cos(angle_rad))
        y_end = 1.00 * np.sin(angle_rad) + small_y_offsets[small_idx]
        small_idx += 1

        ha = 'left' if np.cos(angle_rad) > 0 else 'right'
        connectionstyle = f"angle,angleA=0,angleB={angle}"

        ax.annotate(
            text_str,
            xy=(x_start, y_start),
            xytext=(x_end, y_end),
            ha=ha,
            va='center',
            fontsize=9.5,
            fontweight='bold',
            color='#3a3a3a',
            family='DejaVu Sans',
            arrowprops=dict(
                arrowstyle='-',
                color='#888888',
                lw=0.8,
                shrinkA=0,
                shrinkB=2,
                connectionstyle=connectionstyle
            )
        )

# 右侧图例：放松一点、字体大一点、不要太挤
legend = ax.legend(
    wedges,
    labels,
    loc='upper right',
    bbox_to_anchor=(1.05, 1),   # 靠近图，但不贴太紧
    ncol=1,
    fontsize=10.5,                # 图例字体变大
    frameon=True,
    fancybox=True,
    borderpad=0.5,
    labelspacing=0.9,             # 增大上下间距
    handlelength=1.2,
    handletextpad=0.6
)

frame = legend.get_frame()
frame.set_facecolor('#f7f7f7')
frame.set_edgecolor('#bfbfbf')
frame.set_linewidth(0.6)

ax.axis('equal')

# 关键：整体更紧凑，减少四周空白
plt.subplots_adjust(
    left=0.04,
    right=0.80,
    top=0.95,
    bottom=0.08
)

plt.savefig('expert_routing_valid_topK_3.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.show()