
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import open3d as o3d
from scene.ptgs.shen_partation_densify import Partition  # 确保该模块路径正确

# xy平面分区
def balanced_binary_partition(points, bounds, threshold, depth=0, max_depth=10, index_prefix=""):
    """
    基于点数的平衡二叉空间分割
    :param points: 点云坐标数组 (N, 3)
    :param bounds: 当前分区的边界 (shapely Polygon)
    :param threshold: 目标分区点数
    :param depth: 当前递归深度
    :param max_depth: 最大递归深度
    :param index_prefix: 分区的索引前缀
    :return: 分区结果列表，每个元素是 Partition 类的实例
    """
    xmin, ymin, xmax, ymax = bounds.bounds
    # 筛选当前区域的点
    in_region = (points[:, 0] >= xmin) & (points[:, 0] <= xmax) & \
                (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
    region_points = points[in_region]
    num_points = len(region_points)

    # 输出当前分区的点数量
    partition_id = index_prefix if index_prefix else "0"
    print(f"分区 {partition_id} 点数量: {num_points}")

    # 判断是否满足分区条件
    if (num_points <= threshold * 1.2) or (depth >= max_depth):
        partition_instance = Partition(
            partition_id=partition_id,
            camera=[],  # 初始化为空列表，因为在这个阶段我们还没有相机信息
            point_cloud=None,  # 假设点云信息可以从其他地方提供
            origin_box=bounds,  # 分区的边界
            point_num=num_points
        )
        return [partition_instance]

    # 如果点数超过上限，进行二叉分割
    if num_points > (threshold * 1.2):
        # 选择切分轴：x 或 y 轴
        x_range = xmax - xmin
        y_range = ymax - ymin
        if x_range >= y_range:
            axis = 0  # x 轴
        else:
            axis = 1  # y 轴

        # 找到切分点，使得左右两部分点数尽量接近一半
        sorted_indices = np.argsort(region_points[:, axis])
        sorted_points = region_points[sorted_indices]
        split_index = num_points // 2
        split_value = sorted_points[split_index, axis]

        # 定义子区域
        if axis == 0:
            left_bounds = Polygon([(xmin, ymin), (split_value, ymin), (split_value, ymax), (xmin, ymax)])
            right_bounds = Polygon([(split_value, ymin), (xmax, ymin), (xmax, ymax), (split_value, ymax)])
        else:
            left_bounds = Polygon([(xmin, ymin), (xmax, ymin), (xmax, split_value), (xmin, split_value)])
            right_bounds = Polygon([(xmin, split_value), (xmax, split_value), (xmax, ymax), (xmin, ymax)])

        # 递归分区
        partitions = []
        # 为子区域分配新的索引前缀，使用 '0' 和 '1' 以保持二叉分割的索引格式
        partitions.extend(
            balanced_binary_partition(
                region_points,
                left_bounds,
                threshold,
                depth + 1,
                max_depth,
                f"{index_prefix}0" if index_prefix else "0"
            )
        )
        partitions.extend(
            balanced_binary_partition(
                region_points,
                right_bounds,
                threshold,
                depth + 1,
                max_depth,
                f"{index_prefix}1" if index_prefix else "1"
            )
        )
        return partitions
    else:
        # 点数不足但未达到最大深度，选择不分割
        partition_instance = Partition(
            partition_id=partition_id,
            camera=[],  # 初始化为空列表，因为在这个阶段我们还没有相机信息
            point_cloud=None,  # 假设点云信息可以从其他地方提供
            origin_box=bounds,  # 分区的边界
            point_num=num_points
        )
        return [partition_instance]


def plot_partitions(points, partitions, output_file, base_threshold):
    """
    可视化分区并保存图片，并为所有分区添加图例
    :param points: 点云坐标 (N, 3)
    :param partitions: 分区列表，每个元素是 Partition 类的实例
    :param output_file: 图片保存路径
    :param base_threshold: 基准分区阈值，用于标签显示条件
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制点云（下采样以提高绘图速度）
    sample_size = min(len(points), 100000)
    sample_indices = np.random.choice(len(points), size=sample_size, replace=False)
    sampled_points = points[sample_indices]
    xs, ys = sampled_points[:, 0], sampled_points[:, 1]
    ax.scatter(xs, ys, s=1, c="blue", alpha=0.3, label="Point Cloud")

    # 使用系统化的颜色映射
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % cmap.N) for i in range(len(partitions))]

    # 绘制每个分区并创建图例项
    for i, partition in enumerate(partitions):
        polygon = partition.origin_box
        count = partition.point_num
        x, y = polygon.exterior.xy
        ax.fill(x, y, alpha=0.3, color=colors[i], edgecolor='k',
                label=f"Partition {partition.partition_id}: {count} pts")
        # 添加标签（可选）
        if count > base_threshold * 0.5:
            ax.text(np.mean(x), np.mean(y), str(count), color="black", fontsize=6, ha='center')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Balanced Binary Space Partitioning Point Cloud")

    # 创建图例，确保所有分区都在图例中显示
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=6, markerscale=0.5)

    plt.savefig(output_file, dpi=300)
    print(f"分区图片已保存到 {output_file}")
    plt.close()


def density_partition(points, bounds, threshold, depth=0):
    """
    基于点数的平衡二叉空间分割分区函数
    :param points: 点云坐标数组 (N, 3)
    :param bounds: 分区的初始边界 (shapely Polygon)
    :param threshold: 目标分区点数
    :param depth: 当前递归深度（默认为0）
    :return: 分区结果列表，每个元素是 Partition 类的实例
    """
    # 直接进行平衡二叉空间分割，无需点密度调整
    partitions = balanced_binary_partition(
        points,
        bounds,
        threshold=threshold,
        depth=depth,
        max_depth=10,
        index_prefix=""
    )
    return partitions

# # #xz平面
# #
# def balanced_binary_partition(points, bounds, threshold, depth=0, max_depth=10, index_prefix=""):
#     """
#     基于点数的平衡二叉空间分割（在 XZ 平面进行分区）
#     :param points: 点云坐标数组 (N, 3)
#     :param bounds: 当前分区的边界 (shapely Polygon)
#     :param threshold: 目标分区点数
#     :param depth: 当前递归深度
#     :param max_depth: 最大递归深度
#     :param index_prefix: 分区的索引前缀
#     :return: 分区结果列表，每个元素是 Partition 类的实例
#     """
#     xmin, zmin, xmax, zmax = bounds.bounds  # 修改为 XZ 平面的边界
#     # 筛选当前区域的点
#     in_region = (points[:, 0] >= xmin) & (points[:, 2] >= zmin) & \
#                 (points[:, 0] <= xmax) & (points[:, 2] <= zmax)
#     region_points = points[in_region]
#     num_points = len(region_points)

#     # 输出当前分区的点数量
#     partition_id = index_prefix if index_prefix else "0"
#     print(f"分区 {partition_id} 点数量: {num_points}")

#     # 判断是否满足分区条件
#     if (num_points <= threshold * 1.2) or (depth >= max_depth):
#         partition_instance = Partition(
#             partition_id=partition_id,
#             camera=[],  # 初始化为空列表，因为在这个阶段我们还没有相机信息
#             point_cloud=None,  # 假设点云信息可以从其他地方提供
#             origin_box=bounds,  # 分区的边界
#             point_num=num_points
#         )
#         return [partition_instance]

#     # 如果点数超过上限，进行二叉分割
#     if num_points > (threshold * 1.2):
#         # 选择切分轴：x 或 z 轴
#         x_range = xmax - xmin
#         z_range = zmax - zmin
#         if x_range >= z_range:
#             axis = 0  # x 轴
#         else:
#             axis = 2  # z 轴

#         # 找到切分点，使得左右两部分点数尽量接近一半
#         sorted_indices = np.argsort(region_points[:, axis])
#         sorted_points = region_points[sorted_indices]
#         split_index = num_points // 2
#         split_value = sorted_points[split_index, axis]

#         # 定义子区域
#         if axis == 0:
#             left_bounds = Polygon([(xmin, zmin), (split_value, zmin), (split_value, zmax), (xmin, zmax)])
#             right_bounds = Polygon([(split_value, zmin), (xmax, zmin), (xmax, zmax), (split_value, zmax)])
#         else:
#             left_bounds = Polygon([(xmin, zmin), (xmax, zmin), (xmax, split_value), (xmin, split_value)])
#             right_bounds = Polygon([(xmin, split_value), (xmax, split_value), (xmax, zmax), (xmin, zmax)])

#         # 递归分区
#         partitions = []
#         # 为子区域分配新的索引前缀，使用 '0' 和 '1' 以保持二叉分割的索引格式
#         partitions.extend(
#             balanced_binary_partition(
#                 region_points,
#                 left_bounds,
#                 threshold,
#                 depth + 1,
#                 max_depth,
#                 f"{index_prefix}0" if index_prefix else "0"
#             )
#         )
#         partitions.extend(
#             balanced_binary_partition(
#                 region_points,
#                 right_bounds,
#                 threshold,
#                 depth + 1,
#                 max_depth,
#                 f"{index_prefix}1" if index_prefix else "1"
#             )
#         )
#         return partitions
#     else:
#         # 点数不足但未达到最大深度，选择不分割
#         partition_instance = Partition(
#             partition_id=partition_id,
#             camera=[],  # 初始化为空列表，因为在这个阶段我们还没有相机信息
#             point_cloud=None,  # 假设点云信息可以从其他地方提供
#             origin_box=bounds,  # 分区的边界
#             point_num=num_points
#         )
#         return [partition_instance]
# import numpy as np
# import open3d as o3d
# from shapely.geometry import Polygon
# from typing import List
# from collections import namedtuple
#
#
#
# def area_based_partition(points, bounds, x_partitions=2, z_partitions=3):
#     """
#     按照 XZ 平面上的包围盒进行等面积划分，不考虑点数分布。
#
#     :param points: 点云坐标数组 (N, 3)
#     :param bounds: 初始分区的边界 (shapely Polygon)
#     :param x_partitions: X 方向分区数
#     :param z_partitions: Z 方向分区数
#     :return: 分区结果列表，每个元素是 Partition 类的实例
#     """
#     xmin, ymin, xmax, ymax = bounds.bounds  # 这里 ymin, ymax 会被忽略，因为只在 XZ 平面分区
#     zmin, zmax = ymin, ymax  # 把 ymin, ymax 作为 zmin, zmax 处理
#
#     x_step = (xmax - xmin) / x_partitions  # X 方向的步长
#     z_step = (zmax - zmin) / z_partitions  # Z 方向的步长
#
#     partitions = []
#
#     for i in range(x_partitions):
#         for j in range(z_partitions):
#             # 计算当前分区的边界
#             x0 = xmin + i * x_step
#             x1 = xmin + (i + 1) * x_step
#             z0 = zmin + j * z_step
#             z1 = zmin + (j + 1) * z_step
#
#             # 定义分区的 Polygon
#             partition_bounds = Polygon([(x0, z0), (x1, z0), (x1, z1), (x0, z1)])
#
#             # 筛选点云中落入此分区的点
#             in_region = (points[:, 0] >= x0) & (points[:, 0] <= x1) & \
#                         (points[:, 2] >= z0) & (points[:, 2] <= z1)
#             region_points = points[in_region]
#
#             # 生成新的 Partition 实例
#             partition_instance = Partition(
#                 partition_id=f"{i}_{j}",  # 分区编号，例如 "0_0", "1_2"
#                 camera=[],  # 初始化为空列表
#                 point_cloud=None,
#                 origin_box=partition_bounds,  # 分区的边界
#                 point_num=len(region_points)
#             )
#
#             partitions.append(partition_instance)
#
#     return partitions
