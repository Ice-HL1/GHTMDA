import os

import matplotlib.pyplot as plt
import numpy as np

# 数据
models = ['GHTMDA_AVG', 'GHTMDA w/o GT', 'GHTMDA w/o HET', 'GHTMDA w/o CVCL','GHTMDA w/o CMCL','GHTMDA w/o CL','GHTMDA']
auc_scores = [0.9606, 0.9777, 0.9478,0.9876,0.9813,0.9807,0.9885]
aupr_scores = [0.9519, 0.9784, 0.9412,0.9879,0.9820,0.9811,0.9887]

# 颜色
colors = ['#9F8DB8', '#7DA494', '#EAB67A', '#E5A79A', '#ABC8E5', '#D8A0C1','#C16E71']

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 自定义Y轴刻度函数
def custom_y_axis(ax):
    ax.set_ylim(0.92, 1.024)
    ax.set_yticks(np.arange(0.92, 1.01, 0.02))
    # ax.spines['top'].set_visible(False)
    ax.spines['top'].set_visible(True)

# AUC 图表
bars1 = ax1.bar(range(len(models)), auc_scores, color=colors)
custom_y_axis(ax1)
ax1.set_ylabel('AUC')
ax1.set_title('(A) AUC Scores')

# AUPR 图表
bars2 = ax2.bar(range(len(models)), aupr_scores, color=colors)
custom_y_axis(ax2)
ax2.set_ylabel('AUPR')
ax2.set_title('(B) AUPR Scores')

# 在柱子上添加数值标签
# 调整数值标签的位置
def add_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,  # 减小标签的垂直偏移
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)
add_labels(ax1, bars1)
add_labels(ax2, bars2)

# 移除x轴标签
ax1.set_xticks([])
ax2.set_xticks([])

# 添加图例（右上角）
def add_legend(ax):
    handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
    ax.legend(handles, models, loc='upper right', bbox_to_anchor=(1, 1),
              ncol=1, fontsize=6)

add_legend(ax1)
add_legend(ax2)

plt.tight_layout()
plt.savefig(os.path.join('result', 'GHTMDA', 'table3.png'))
plt.show()
