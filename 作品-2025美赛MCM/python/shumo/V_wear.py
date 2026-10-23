import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg')

# 定义参数
x = np.linspace(0, 8, 100)  # x轴范围
y_values = [1, 2, 3]  # 不同的y值
vertical_offsets = [0, 20, 40]  # 各曲线的垂直偏移量

# 计算各曲线数据
curve_data = []
for offset in vertical_offsets:
    z = (x - 4) ** 3 + 8 + offset  # 添加垂直偏移
    curve_data.append(z)

# 计算全局最小值
all_z = np.concatenate(curve_data)
global_z_min = np.min(all_z)
global_z_max = np.max(all_z)

# 创建三维图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = ['#FF5733', '#33FF57', '#3357FF']  # 配色方案

# 绘制曲线和填充区域
for i, (y, z) in enumerate(zip(y_values, curve_data)):
    # 绘制曲线
    ax.plot(x, np.full_like(x, y), z,
            color=colors[i],
            linewidth=2,
            label=f'Material {i + 1}')

    # 创建填充曲面（从全局最小值到曲线）
    X = np.vstack([x, x])  # X坐标网格
    Y = np.vstack([np.full_like(x, y), np.full_like(x, y)])  # Y坐标网格
    Z = np.vstack([np.full_like(x, global_z_min), z])  # Z坐标网格

    # 绘制填充曲面
    ax.plot_surface(X, Y, Z,
                    color=colors[i],
                    alpha=0.3,
                    antialiased=True)

# 可视化设置
ax.set_xlabel('Time (t)')
ax.set_ylabel('Material Type')
ax.set_zlabel('Wear Degree')

# 调整坐标轴范围
ax.set_xlim(0, 8)
ax.set_ylim(0, 4)
ax.set_zlim(global_z_min, global_z_max)

# 移除坐标轴刻度
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# 添加材质标签
label_positions = [(8, 1, 5), (8, 2, 5), (8, 3, 5)]
labels = ['limestone', 'marble', 'granite']
for pos, label in zip(label_positions, labels):
    ax.text(*pos, label,
            color='black',
            fontsize=12,
            ha='right')

# 优化视角
ax.view_init(elev=25, azim=-45)  # 调整视角角度

plt.tight_layout()
plt.savefig('wear_analysis.png', dpi=500, bbox_inches='tight')
plt.show()