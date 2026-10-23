import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
import pandas as pd

# 定义台阶的尺寸
length = 350  # 长度 (mm)
width = 150  # 宽度 (mm)
max_depth = 40  # 深度最大值 (mm)
num_footprints = 100  # 脚印数量

# 创建一个网格
x = np.arange(0, length, 0.2)  # 长度为 350mm, 步长为 1mm
y = np.arange(0, width, 0.2)  # 宽度为 150mm, 步长为 1mm
X, Y = np.meshgrid(x, y)  # 创建长宽网格


# 定义深度计算函数
def calculate_depth(X, Y, num_footprints):
    depths = np.random.rand(*X.shape) * (max_depth / 4)  # 基础噪声
    center_x = length / 2
    center_y = width / 8

    for _ in range(num_footprints):
        if np.random.rand() < 0.7:
            center_x_pos = np.random.uniform(center_x - 30, center_x + 30)
            center_y_pos = np.random.uniform(center_y - 15, center_y + 15)
        else:
            center_x_pos = np.random.uniform(0, length)
            center_y_pos = np.random.uniform(0, width)

        max_depth_for_footprint = np.random.uniform(15, max_depth)
        sigma_x = np.random.uniform(5, 15)
        sigma_y = np.random.uniform(5, 15)

        footprint_depth = max_depth_for_footprint * np.exp(-(((X - center_x_pos) ** 2) / (2 * sigma_x ** 2) +
                                                             ((Y - center_y_pos) ** 2) / (2 * sigma_y ** 2)))
        depths += footprint_depth

    depths = np.clip(depths, 0, max_depth)
    return depths


# 计算深度数据
depths = calculate_depth(X, Y, num_footprints)

data = {
    'X (mm)': X.flatten(),
    'Y (mm)': Y.flatten(),
    'Depth (mm)': depths.flatten()
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 添加说明行
description = "此表格展示了每个 (x, y) 坐标对应的深度值，单位为毫米。"

# 保存到 CSV 文件
with open('depth_data_with_description.csv', 'w', encoding='utf-8') as f:
    # 写入说明
    f.write(description + '\n')  # 添加说明并换行
    # 写入 DataFrame 的头部和数据
    df.to_csv(f, index=False)  # 将数据添加到文件中


# 绘制平面热力图
plt.figure(figsize=(10, 6), dpi=75)
heatmap = plt.imshow(depths, extent=[0, length, 0, width], origin='lower', cmap='viridis')
plt.colorbar(heatmap, label='Depth (mm)')
plt.title('Natural Depth Distribution of Worn Step with Footprints')
plt.xlabel('Length (mm)')
plt.ylabel('Width (mm)')
plt.grid(False)
plt.xlim(0, length)
plt.ylim(0, width)
plt.show()

# 绘制3D柱状图
fig = plt.figure(figsize=(12, 8),dpi=75)
ax = fig.add_subplot(111, projection='3d')

# 创建柱状图数据
x_pos = X.flatten()
y_pos = Y.flatten()
z_pos = np.zeros_like(depths.flatten())

# 每个柱子的宽度和深度
dx = dy = 0.2  # 根据网格调整宽度
dz = depths.flatten()

# 为每个柱子生成颜色，使用与热力图相同的颜色映射
norm = Normalize(vmin=0, vmax=max_depth)
colors = cm.viridis(norm(dz))

# 绘制柱状图
ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, zsort='average')

# 设置轴标签
ax.set_xlabel('Length (mm)')
ax.set_ylabel('Width (mm)')
ax.set_zlabel('Depth (mm)')
ax.set_title('3D Depth Distribution of Worn Step with Footprints')

# 添加颜色条，与热力图一致
mappable = cm.ScalarMappable(norm=norm, cmap='viridis')
mappable.set_array(dz)
cbar = fig.colorbar(mappable, ax=ax, label='Depth (mm)')

# 显示图形
plt.show()
