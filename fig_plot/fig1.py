import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# 创建一个画布
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
fig.patch.set_facecolor('lightgray') 
ax.set_facecolor('lightgray')  # 设置坐标轴区域的背景颜色
# 数据点（紫色圆点） - 增加点云的数量，使其更密集
# 假设点云的范围为(0, 1)，并且模拟的点云相对集中在中心
x = np.random.normal(0.45, 0.2, 200)  # 增加点云数量并减少标准差，使其更加集中
y = np.random.normal(0.45, 0.2, 200)  # 增加点云数量并减少标准差
points = np.column_stack((x, y))

# 摄像机位置（黄色三角形） - 放置在点云的外围
# 假设相机在一个均匀网格上排列，围绕点云放置
camera_x = np.linspace(0.3, 0.6, 5)  # 8个相机，沿x轴均匀分布
camera_y = np.linspace(0.3, 0.6, 5)  # 8个相机，沿y轴均匀分布
cameras = np.array(np.meshgrid(camera_x, camera_y)).T.reshape(-1, 2)

# 绘制所有点云（紫色），设置较大的点云大小
ax.scatter(points[:, 0], points[:, 1], color='purple', label='Point Cloud', s=100, alpha=0.9, zorder=1)

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
plt.savefig('fig1.png', dpi=300, bbox_inches='tight')
# 显示图形
# plt.show()
