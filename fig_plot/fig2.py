import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

np.random.seed(42)

# 创建一个画布
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
fig.patch.set_facecolor('lightgray') 
ax.set_facecolor('lightgray')  # 设置坐标轴区域的背景颜色

# 数据点（紫色圆点） - 增加点云的数量，使其更密集
x = np.random.normal(0.45, 0.2, 200)  # 增加点云数量并减少标准差，使其更加集中
y = np.random.normal(0.45, 0.2, 200)  # 增加点云数量并减少标准差
points = np.column_stack((x, y))

# 设置矩形框
rect = patches.Rectangle((0.2, 0.2), 0.5, 0.5, linewidth=2, edgecolor='black', facecolor='none', alpha=0.5)
# 添加矩形到坐标轴
ax.add_patch(rect)

# 判断哪些点在矩形框内
mask_inside = (points[:, 0] >= 0.2) & (points[:, 0] <= 0.7) & (points[:, 1] >= 0.2) & (points[:, 1] <= 0.7)
mask_outside = ~mask_inside  # 不在矩形框内的点

# 绘制所有点云（紫色），设置较大的点云大小
ax.scatter(points[mask_inside, 0], points[mask_inside, 1], color='purple', label='Point Cloud', s=100, alpha=0.9, zorder=1)
ax.scatter(points[mask_outside, 0], points[mask_outside, 1], color='gray', label='Outside Points', s=100, alpha=0.9, zorder=1)

# 摄像机位置（黄色三角形），设置较小的相机大小
camera_x = np.linspace(0.3, 0.6, 5)  # 8个相机，沿x轴均匀分布
camera_y = np.linspace(0.3, 0.6, 5)  # 8个相机，沿y轴均匀分布
cameras = np.array(np.meshgrid(camera_x, camera_y)).T.reshape(-1, 2)

# 绘制所有相机（黄色三角形），设置较小的相机大小
ax.scatter(cameras[:, 0], cameras[:, 1], color='yellow', marker='^', label='Camera', s=600, alpha=0.9, edgecolors='black', linewidth=1.5, zorder=2)

# 设置坐标轴
# ax.set_xlim([0.1, 0.9])
# ax.set_ylim([0.1, 0.9])
# ax.set_aspect('equal')

# 设置背景颜色
ax.set_facecolor('white')  # 白色背景

# 去除网格和坐标轴线
ax.grid(False)
ax.set_xticks([]) 
ax.set_yticks([])

# 显示图例
# ax.legend()

# 保存图像
plt.savefig('fig2.png', dpi=300, bbox_inches='tight')

# 显示图形
# plt.show()
