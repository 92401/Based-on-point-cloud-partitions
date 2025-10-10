import math
import os
import pickle

import open3d as o3d
import random
import matplotlib.pyplot as plt
import numpy as np
import copy
import torch
from shapely.vectorized import contains
from shapely.strtree import STRtree
from scene.ptgs.graham_scan import run_graham_scan
from shapely.geometry import Polygon, LineString,Point
from trimesh.path.packing import images
from typing import NamedTuple, List, Optional

from scene.ptgs.save_partition import save_partition_data
from shen_data_read import partition, CameraInfo
from utils.camera_utils import cameraList_from_camInfos_partition


class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

class Partition(NamedTuple):
    partition_id: str  # 分区的字符编号
    camera: list  # 该分区所有相机的pose
    point_cloud: BasicPointCloud
    origin_box: Polygon  # 对应的边界 Polygon 实例,分区点云边界
    point_num: int  # 该分区包含的点数量
    extend_rate: float = 0  # 该分区的拓展率，默认为0
    extend_box: Optional[Polygon] = None  # 动态拓展后的点云边界，默认为None
class CameraPose(NamedTuple):
    camera: CameraInfo
    pose: np.array  # [x, y, z] 坐标
#导入数据
def quad_tree_partition(points, bounds, threshold, depth=0, index_prefix=""):
    """
    递归实现四叉树分区
    :param points: 点云的坐标数组
    :param bounds: 当前分区的边界 Polygon 实例
    :param threshold: 每个分区允许的最大点数
    :param depth: 当前分区的深度
    :param index_prefix: 当前分区的索引前缀
    :return: 分区结果列表，每个元素是 Partition 类的实例
    """
    # 筛选出当前区域的点
    xmin, ymin, xmax, ymax = bounds.bounds  # 使用 Polygon 获取边界坐标
    in_region = (points[:, 0] >= xmin) & (points[:, 0] <= xmax) & (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
    region_points = points[in_region]
    # 输出当前分区的点数量
    print(f"分区 {index_prefix} 点数量: {len(region_points)}")
    # 如果点数量小于阈值，返回当前分区（创建 Partition 实例）
    if len(region_points) <= threshold:
        partition_instance = Partition(
            partition_id=index_prefix,
            camera=[],  # 初始化为空列表，因为在这个阶段我们还没有相机信息
            point_cloud=None,  # 假设 BasicPointCloud 可以从点列表创建
            origin_box=Polygon(bounds),  # 将 bounds 转换为 Polygon 对象
            point_num=len(region_points),
            # extend_rate 和 extend_box 使用默认值
        )
        return [partition_instance]
    # 否则继续分割为 4 个子区域
    x_mid = (xmin + xmax) / 2
    y_mid = (ymin + ymax) / 2
    sub_bounds = [
        Polygon([(xmin, ymin), (x_mid, ymin), (x_mid, y_mid), (xmin, y_mid)]),  # 左下
        Polygon([(x_mid, ymin), (xmax, ymin), (xmax, y_mid), (x_mid, y_mid)]),  # 右下
        Polygon([(xmin, y_mid), (x_mid, y_mid), (x_mid, ymax), (xmin, ymax)]),  # 左上
        Polygon([(x_mid, y_mid), (xmax, y_mid), (xmax, ymax), (x_mid, ymax)])  # 右上
    ]
    partitions = []
    for i, sub_bound in enumerate(sub_bounds):
        sub_index = f"{index_prefix}{i}"
        # 获取子区域的分区结果
        sub_partitions = quad_tree_partition(region_points, sub_bound, threshold, depth + 1, sub_index)
        # 确保返回值是一个列表，将其扩展到当前分区列表中
        partitions.extend(sub_partitions)
    return partitions

def recursive_merge(partition, partitions,threshold_value,skip_partition):
         # 查找邻接区域
        adjacent_partitions = find_adjacent_partitions(partition, partitions,skip_partition)
        if adjacent_partitions:
             # 先找到邻接区域中点数最少的区域
             min_adj_partition = min(adjacent_partitions, key=lambda x: x.point_num)
             min_partition_id = min_adj_partition.partition_id
             # 合并当前区域和点数最少的邻接区域
             new_partition = merge_partitions(partition, min_adj_partition)
             # 如果合并后的区域点数超出了阈值，则不再继续合并
             if new_partition.point_num > threshold_value:
                 print(
                     f"合并后的区域点数超过阈值 {threshold_value}, 不再合并区域 {partition.partition_id} 和 {min_partition_id}")
                 return partition  # 保留原区域
             else:
                 # 更新当前区域为合并后的区域
                 print(
                     f"区域 {partition.partition_id} 和 {min_partition_id} 合并后的点数为 {new_partition.point_num}")
                 skip_partition.add(min_partition_id)
                 return new_partition
def filter_partitions_by_point_count(partitions, threshold_value, min_threshold=10):
    """
    根据每个分区的点数量进行过滤，并处理点数过少的区域。
    小于 min_threshold 的区域将被删除；小于均值减一倍标准差的区域将与邻接区域合并。

    :param partitions: 分区数据，每个分区的数据是 (分区编号, 边界, 点数量)
                       边界格式: [min_x, max_x, min_z, max_z]
    :param threshold_value: 最大点数阈值，如果合并后的点数超过该值，则不继续合并
    :param min_threshold: 最小点数阈值，点数小于该值的区域将被删除
    :return: 剔除并合并后的分区列表
    """
    # 过滤掉点数小于 min_threshold 的分区
    partitions = [p for p in partitions if p.point_num >= min_threshold]
    # 统计每个分区的点数量
    partition_sizes = [partition.point_num for partition in partitions]
    # 计算点数的均值和标准差
    mean_size = np.mean(partition_sizes)
    std_size = np.std(partition_sizes)
    # 计算小于均值减一倍标准差的阈值
    lower_threshold = mean_size
    print(f"点数量的均值: {mean_size}, 标准差: {std_size}")
    print(f"过滤范围: 点数 >= {min_threshold}, 点数 >= {lower_threshold} 参与合并处理")
    # 用于存储处理后的分区
    merged_partitions = []
    skip_partition = set()
    for partition in partitions:
        partition_id = partition.partition_id
        if partition_id in skip_partition:
            continue
        count = partition.point_num
        if count < lower_threshold:
            skip_partition.add(partition_id)
            print(f"区域 {partition_id} 点数小于均值 ({lower_threshold}), 尝试合并")
            adjacent_partitions = find_adjacent_partitions(partition, partitions,skip_partition)
            if not adjacent_partitions:
                print(f"区域 {partition_id} 无可合并的邻接区域")
                merged_partitions.append(partition)
                continue
            min_adj_partition = min(adjacent_partitions, key=lambda x: x.point_num)
            min_partition_id = min_adj_partition.partition_id
            skip_partition.add(min_partition_id)
            print(f"区域 {partition_id} 的最小点数邻域是({min_partition_id}), 尝试合并")
            new_partition = merge_partitions(partition, min_adj_partition)
            if new_partition.point_num < lower_threshold:
                print(
                    f"合并后的区域点数依然小于阈值 {mean_size}, 继续合并区域  {new_partition.partition_id}")
                partitions = [p for p in partitions if p.partition_id not in skip_partition]
                new_partition= recursive_merge(new_partition, partitions,threshold_value,skip_partition)

            if new_partition.point_num > threshold_value:
                print(
                    f"合并后的区域点数超过阈值 {threshold_value}, 不再合并区域 {partition.partition_id} 和 {min_partition_id}")
                merged_partitions.append(partition)
            print(f"区域 {partition.partition_id} 和 {min_partition_id} 合并后的区域id为{new_partition.partition_id}点数为 {new_partition.point_num}")
            partition = new_partition
        merged_partitions.append(partition)
    print(skip_partition)
    merged_partitions = [p for p in merged_partitions if p.partition_id not in skip_partition]
    return merged_partitions

def find_adjacent_partitions(partition, partitions,skip_partition):
    """
    寻找给定分区的所有相邻分区
    :param partition: 当前分区对象（包含 bounds 信息）
    :param partitions: 所有分区的列表
    :return: 相邻分区的列表
    """
    bounds = partition.origin_box  # 当前分区的边界
    adjacent_partitions = []
    for adj_partition in partitions:
        adj_bounds = adj_partition.origin_box  # 邻接分区的边界
        if adj_partition.partition_id in skip_partition:
            continue
        # 忽略自身
        if bounds == adj_bounds:
            continue
        # 计算交集
        intersection = bounds.intersection(adj_bounds)
        # 判断是否存在非零长度的公共边界
        if isinstance(intersection, LineString) and intersection.length > 0:
            adjacent_partitions.append(adj_partition)
    return adjacent_partitions

def merge_partitions(partition1, partition2):
    """
    合并两个区域的边界，返回拼接后的区域对象。
    :param partition1: 第一个分区的 Partition 对象
    :param partition2: 第二个分区的 Partition 对象
    :return: 合并后的分区对象
    """
    # 提取每个分区的数据
    partition1_id = partition1.partition_id
    partition1_bounds = partition1.origin_box
    partition1_count = partition1.point_num
    partition2_id = partition2.partition_id
    partition2_bounds = partition2.origin_box
    partition2_count = partition2.point_num
    # 合并两个多边形
    merged_polygon = partition1_bounds.union(partition2_bounds)
    # 合并后的点数
    merged_count = partition1_count + partition2_count
    # 合并后的区域 ID（格式为 "partition1_id-partition2_id"）
    merged_id = f"{partition1_id}-{partition2_id}"
    # 创建并返回合并后的 Partition 对象
    # 创建并返回合并后的 Partition 对象
    merged_partition = Partition(
        partition_id=merged_id,
        camera=[],
        point_cloud=None,
        origin_box=merged_polygon,
        point_num=merged_count
        # extend_rate 和 extend_box 使用默认值
    )
    return merged_partition


def is_rectangle(polygon):
    """
    判断一个 Shapely 多边形是否是矩形
    :param polygon: Shapely 多边形对象
    :return: 布尔值，表示是否为矩形
    """
    minx, miny, maxx, maxy = polygon.bounds
    rectangle = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])
    return polygon.equals(rectangle)


def expand_partitions(filtered_partitions,point3D ,base_expansion_rate=0.2, base_density=None):
    def extract_point_cloud(pcd, polygon):
        """
        根据给定的多边形从点云数据中筛选点
        :param pcd: 点云对象，包含点坐标、颜色和法向量
        :param polygon_coords: 多边形的顶点坐标，列表形式 [(x1, y1), (x2, y2), ...]
        :return: 筛选后的点坐标、颜色和法向量
        """
        # 创建多边形的空间索引（R-tree）
        tree = STRtree([polygon])
        # 将点云坐标转化为 NumPy 数组
        points = np.array(pcd.points)
        # 使用 NumPy 向量化操作：批量检查点是否在多边形内
        mask = contains(polygon, points[:, 0], points[:, 1])
        # 使用掩码筛选点云数据
        filtered_points = points[mask]
        filtered_colors = np.array(pcd.colors)[mask]
        filtered_normals = np.array(pcd.normals)[mask]
        return filtered_points, filtered_colors, filtered_normals
    expanded_partitions = []
    for partition in filtered_partitions:
        expansion_rate = base_expansion_rate
        # 计算扩展距离
        # 使用多边形的最小外接矩形的对角线长度作为参考
        minx, miny, maxx, maxy = partition.origin_box.bounds
        diagonal = ((maxx - minx) ** 2 + (maxy - miny) ** 2) ** 0.5
        expansion_distance = diagonal * expansion_rate / 2  # 除以2是为了得到类似于之前的扩展效果
        # 使用buffer方法扩展多边形
        expanded_bounds = partition.origin_box.buffer(expansion_distance, join_style=2)
        points, colors, normals = extract_point_cloud(point3D,expanded_bounds)
        pcd_i=BasicPointCloud(points, colors, normals)   #这里不知道循环时候是否会清空上一次的
        # 创建新的扩展后的分区
        expanded_partition = Partition(
            partition_id=partition.partition_id,
            origin_box=partition.origin_box,
            camera=[],
            extend_rate=expansion_rate,
            extend_box=expanded_bounds,
            point_num=partition.point_num,
            point_cloud=pcd_i)
        expanded_partitions.append(expanded_partition)
        print(f"分区 {partition.partition_id} - 扩展率: {expansion_rate:.2f}")
    return expanded_partitions

def plot_quad_tree(partitions, points, picname):  #适用于不加入相机的画图
    fig, ax = plt.subplots(figsize=(12, 8))  # 调整图形尺寸
    # 绘制点云
    xs, ys = zip(*points)
    ax.scatter(xs, ys, s=10, c="blue", alpha=0.5)  # 调整点的大小
    # 为每个区域分配一个随机颜色
    colors = [tuple(random.random() for _ in range(3)) for _ in range(len(partitions))]
    # 绘制分区
    for partition in partitions:
        idx = partition.partition_id
        bounds = partition.origin_box
        count = partition.point_num
        color = colors[partitions.index(partition)]  # 根据 partition 的索引选择颜色
        print(f"分区 {idx}，点数量: {count}")
        # 如果 bounds 是 MultiPolygon, 需要遍历其中的每个 Polygon 对象
        if bounds.geom_type == 'MultiPolygon':
            for polygon in bounds.geoms:  # 使用 bounds.geoms 获取内部所有 Polygon
                x, y = polygon.exterior.xy
                # 绘制每个 Polygon 的边界
        elif bounds.geom_type == 'Polygon':
            x, y = bounds.exterior.xy
        ax.fill(x, y, alpha=0.5, color=color)  # 使用填充色彩来显示分区
        # 添加分区编号和点数量，使用与区域相同的颜色
        ax.text(np.mean(x),np.mean(y),f"{idx} ({count})",color=color,fontsize=8,ha='center',va='center' )
    # 设置坐标轴范围（可选）
    ax.set_xlim([min(xs) - 10, max(xs) + 10])
    ax.set_ylim([min(ys) - 10, max(ys) + 10])
    # 标签和标题
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("QuadTree Partition Visualization")
    # 保存图像
    plt.savefig(picname + ".png")


def plot_subregions(partitions, output_dir):
    """
    为每个子区域绘制单独的二维图片，包含点云、边界框和相机位置。
    参数:
    partitions: 包含所有子区域信息的列表
    output_dir: 输出图片的目录
    """
    for partition in partitions:
        fig, ax = plt.subplots(figsize=(10, 10))
        # 处理点云数据
        points = partition.point_cloud
        if points is not None and len(points) > 0:
            if isinstance(points, BasicPointCloud):
                # 如果是 BasicPointCloud 对象，直接访问 points 属性
                points = points.points
            if isinstance(points, torch.Tensor):
                points = points.cpu().numpy()
            elif isinstance(points, tuple):
                points = np.array(points)
            elif not isinstance(points, np.ndarray):
                points = np.array(list(points))
            ax.scatter(points[:, 0], points[:, 1], c='blue', s=1, alpha=0.5)
        # 绘制边界框
        boundary = partition.extend_box
        if isinstance(boundary, Polygon):
            x, y = boundary.exterior.xy
            ax.plot(x, y, color='green', linewidth=2)
        # 绘制相机位置
        cameras = partition.camera
        for camera_i in cameras:
            position = camera_i.pose
            ax.scatter(position[0], position[1], c='red', s=50, marker='^')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Partition ID: {partition.partition_id}')
        ax.set_aspect('equal')
        # 保存图片
        plt.savefig(f'{output_dir}/partition_{partition.partition_id}.png')
        plt.close(fig)


def plot_single_subregion_with_updates(partition, new_cameras, new_points, output_dir):
    """
    为单个子区域绘制二维图片，包含原始和新加入的点云、边界框和相机位置。
    参数:
    partition: 原始分区对象
    new_cameras: 新加入的相机列表
    new_points: 新加入的点云数据
    output_dir: 输出图片的目录
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    # 绘制原始点云
    original_points = partition.point_cloud
    if original_points is not None and len(original_points) > 0:
        if isinstance(original_points, BasicPointCloud):
            original_points = original_points.points
        if isinstance(original_points, torch.Tensor):
            original_points = original_points.cpu().numpy()
        elif isinstance(original_points, tuple):
            original_points = np.array(original_points)
        elif not isinstance(original_points, np.ndarray):
            original_points = np.array(list(original_points))
        ax.scatter(original_points[:, 0], original_points[:, 1], c='blue', s=1, alpha=0.5, label='Original Points')
    # 绘制新加入的点云
    if new_points is not None and len(new_points) > 0:
        all_x = []
        all_y = []
        for point_group in new_points:
            if isinstance(point_group, (list, np.ndarray)) and len(point_group) > 0:
                points_array = np.array(point_group)
                if points_array.ndim == 2 and points_array.shape[1] >= 2:
                    all_x.extend(points_array[:, 0])
                    all_y.extend(points_array[:, 1])
        if all_x and all_y:
            ax.scatter(all_x, all_y, c='green', s=1, alpha=0.5, label='New Points')
    # 绘制边界框
    boundary = partition.extend_box
    if isinstance(boundary, Polygon):
        x, y = boundary.exterior.xy
        ax.plot(x, y, color='black', linewidth=2, label='Boundary')
    # 绘制原始相机位置
    original_cameras = partition.camera
    for camera in original_cameras:
        position = camera.pose
        ax.scatter(position[0], position[1], c='red', s=50, marker='^', label='Original Camera')
    # 绘制新加入的相机位置
    for camera in new_cameras:
        position = camera.pose
        ax.scatter(position[0], position[1], c='orange', s=50, marker='s', label='New Camera')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Partition ID: {partition.partition_id} (Updated)')
    ax.set_aspect('equal')
    # 去除重复的图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    # 保存图片
    plt.savefig(f'{output_dir}/updated_partition_{partition.partition_id}.png')
    plt.close(fig)

def assign_cameras_to_partitions(expanded_partitions: List[Partition], train_cameras: List[CameraInfo]):
    """
    读取相机信息，将相机加入到各个分区中,输入分区信息也就是expanded_partition和训练相机的信息，返回加入了相机信息的分区列表
    1.读取相机的位置信息，根据坐标判断在哪个expand_partition中
    2.将该相机信息加入到该区域中
    将相机信息分配到相应的分区中。
    :param expanded_partitions: 扩展后的分区列表
    :param train_cameras: 训练相机信息列表
    :return: 更新后的分区列表，每个分区包含相应的相机信息
    """
    updated_partitions = []
    CameraPose_list = []
    camera_centers = []
    for idx, camera in enumerate(train_cameras):
        # 确保 camera_center 是 CPU tensor 或 NumPy 数组
        if isinstance(camera.camera_center, torch.Tensor):
            pose = camera.camera_center.cpu().numpy()
        elif isinstance(camera.camera_center, np.ndarray):
            pose = camera.camera_center
        else:
            pose = np.array(camera.camera_center)
        camera_centers.append(pose)
        CameraPose_list.append(CameraPose(camera=camera, pose=pose))
    for partition in expanded_partitions:
        cameras_in_partition = []
        for camera_pose in CameraPose_list:
        # 检查相机是否在当前分区的扩展边界内
            if partition.extend_box.contains(Point(camera_pose.pose[:2])):  #
                cameras_in_partition.append(camera_pose)
        # 创建新的 Partition 实例，包含相机信息
            updated_partition = Partition(
                partition_id=partition.partition_id,
                camera=cameras_in_partition,   #存储的相机pose列表
                point_cloud=partition.point_cloud,
                origin_box=partition.origin_box,
                point_num=partition.point_num,
                extend_rate=partition.extend_rate,
                extend_box=partition.extend_box)
        updated_partitions.append(updated_partition)
        print(f"分区 {updated_partition.partition_id}:")
        print(f"  - 相机数量: {len(updated_partition.camera)}")
        print(f"  - 点云数量: {updated_partition.point_num}")
        print(f"  - 扩展率: {updated_partition.extend_rate}")
        print("  ------------------------------------")
    return updated_partitions



def project_points_to_camera(points, camera, device='cuda'):
    """
    使用GPU（PyTorch）加速版的将点云投影到相机图像平面函数。

    参数:
        points (list或np.ndarray): 形状为 (N,3) 的3D点云。确保已是标准浮点数格式。
        camera (object): 相机对象，需包含下列属性：
            - R: (3,3) np.ndarray 世界到相机的旋转矩阵
            - T: (3,) 或 (3,1) np.ndarray 平移向量
            - FoVx: float 水平视场角(弧度)
            - FoVy: float 垂直视场角(弧度)
            - image_width: int 图像宽度(像素)
            - image_height: int 图像高度(像素)
        device (str): 'cuda'或'cpu'，默认为'cuda'在GPU上加速计算。

    返回:
        projected_points_list (list): 投影到图像平面上的2D点列表，形状为 (M, 2)。
        valid_mask (np.ndarray): 布尔数组，形状为 (N,)，表示每个输入点是否有效（在图像内）。
    """
    # 确保 points 是 np.ndarray
    points = np.asarray(points, dtype=np.float32)
    # 转为Torch张量并放入GPU
    points_t = torch.tensor(points, device=device)
    # 从camera中提取R、T并转为Tensor
    R = torch.tensor(camera.R, dtype=torch.float32, device=device)
    T = torch.tensor(camera.T, dtype=torch.float32, device=device)
    # 若 T 为 (3,1) 则展平为 (3,)
    if T.shape == (3, 1):
        T = T.view(3, )
    # 构建4x4变换矩阵 W2C (world to camera)
    W2C = torch.eye(4, dtype=torch.float32, device=device)
    W2C[:3, :3] = R
    W2C[:3, 3] = T
    # 将点云转换为齐次坐标 (N,4)
    N = points_t.shape[0]
    ones = torch.ones((N, 1), dtype=torch.float32, device=device)
    points_homog = torch.cat((points_t, ones), dim=1)  # (N,4)
    # 世界坐标 -> 相机坐标
    points_camera_homog = (W2C @ points_homog.transpose(0, 1)).transpose(0, 1)  # (N,4)
    points_camera = points_camera_homog[:, :3] / points_camera_homog[:, 3, None]  # (N,3)
    # 过滤掉相机后方的点(Z>0)
    in_front_mask = points_camera[:, 2] > 0
    if torch.sum(in_front_mask) == 0:
        # 无点在前方
        valid_mask = torch.zeros(N, dtype=torch.bool, device=device)
        return [], valid_mask.cpu().numpy()

    # 保留在前方的点
    points_camera_in_front = points_camera[in_front_mask]

    # 计算内参矩阵
    fx = camera.image_width / (2 * math.tan(camera.FoVx / 2))
    fy = camera.image_height / (2 * math.tan(camera.FoVy / 2))
    intrinsic_matrix = torch.tensor([
        [fx, 0, camera.image_width / 2],
        [0, fy, camera.image_height / 2],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    # 投影到图像平面 (M,3)
    points_image_homog = (intrinsic_matrix @ points_camera_in_front.transpose(0, 1)).transpose(0, 1)
    points_image = points_image_homog[:, :2] / points_image_homog[:, 2, None]
    # 检查点是否在图像范围内
    in_image = (points_image[:, 0] >= 0) & (points_image[:, 0] < camera.image_width) & \
               (points_image[:, 1] >= 0) & (points_image[:, 1] < camera.image_height)
    valid_mask = torch.zeros(N, dtype=torch.bool, device=device)
    valid_mask[in_front_mask] = in_image
    # 将结果转回CPU
    projected_points_list = points_image[in_image].cpu().numpy().tolist()
    valid_mask = valid_mask.cpu().numpy()
    return projected_points_list, valid_mask

def run_graham_scan(image_points, image_width, image_height):
    """
    使用Graham Scan算法计算一组2D点的凸包，然后计算凸包面积与图像面积比值。

    参数:
        image_points: (N, 2) numpy数组或可迭代对象，每行是一个点(x, y)
        image_width: 图像宽度(像素)
        image_height: 图像高度(像素)

    返回:
        pkg: dict, 包含:
            "intersection_rate": 凸包面积与图像面积之比 (float)
    """
    points = np.array(image_points, dtype=float)
    # 如果点数不足3个，直接返回0
    if len(points) < 3:
        return {"intersection_rate": 0.0}
    # Graham Scan 算法步骤:
    # 1. 找到y坐标最低的点（若有多个，取x坐标最小的）
    #    此点作为旋转扫描的基点p0
    def polar_angle(p0, p1):
        # 相对于p0的极角
        y_span = p1[1] - p0[1]
        x_span = p1[0] - p0[0]
        return np.arctan2(y_span, x_span)
    def distance_sq(p0, p1):
        # p0与p1的距离平方，用于在极角相同时比较谁更近
        return (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2
    # 找基点p0
    min_y = np.min(points[:, 1])
    candidate = points[points[:, 1] == min_y]
    p0 = candidate[np.argmin(candidate[:, 0])]  # 若多点最低则选x最小的那个
    # 2. 其他点相对p0排序，根据极角，从小到大。如果极角相同，距离近的在前。
    sorted_points = sorted(points, key=lambda p: (polar_angle(p0, p), -distance_sq(p0, p)))
    # 3. 使用栈进行扫描构建凸包
    hull = [p0]
    for pt in sorted_points[1:]:
        # 判断当前点与栈顶两个点的转向
        while len(hull) > 1:
            # hull[-1]为栈顶点, hull[-2]为次栈顶点
            cross = ((hull[-1][0] - hull[-2][0]) * (pt[1] - hull[-2][1]) -
                     (hull[-1][1] - hull[-2][1]) * (pt[0] - hull[-2][0]))
            if cross <= 0:
                # <=0表示非左转(可能右转或共线), 需弹出
                hull.pop()
            else:
                break
        hull.append(pt)
    # 如果凸包点数仍小于3，无法构成多边形
    if len(hull) < 3:
        return {"intersection_rate": 0.0}
    # 4. 使用Shoelace formula计算凸包面积
    hull_arr = np.array(hull)
    x = hull_arr[:, 0]
    y = hull_arr[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    # 计算intersection_rate
    image_area = image_width * image_height
    intersection_rate = area / image_area
    return {"intersection_rate": intersection_rate}
def downsample_point_cloud(pc: BasicPointCloud, voxel_size: float) -> BasicPointCloud:
    """
    对点云进行体素下采样。
    参数：
        pc: 输入点云（BasicPointCloud）
        voxel_size: 体素大小，越大下采样率越高（点数越少）

    返回：
        downsampled_pc: 下采样后的点云（BasicPointCloud）
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive.")

    points = pc.points
    colors = pc.colors
    normals = pc.normals

    # 若点数为0，直接返回
    if len(points) == 0:
        return pc

    # 计算点云边界，用于确定体素坐标
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)

    # 避免除0错误和特别极端情况
    if np.any(np.isclose(voxel_size, 0)):
        return pc

    # 将点的坐标转换为体素索引坐标（整数）
    # voxel_idx是(N, 3)的整数数组，每个点对应所在的voxel grid坐标
    voxel_idx = np.floor((points - min_bound) / voxel_size).astype(np.int32)

    # 使用字典将点根据voxel_idx分组
    # 字典键为3D体素坐标元组，值为对应点的index列表
    voxel_dict = {}
    for i, vid in enumerate(voxel_idx):
        key = (vid[0], vid[1], vid[2])
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(i)

    # 对每个voxel取平均值(点坐标、颜色、法线)
    new_points = []
    new_colors = [] if colors is not None and len(colors) == len(points) else None
    new_normals = [] if normals is not None and len(normals) == len(points) else None

    for key, idx_list in voxel_dict.items():
        selected_points = points[idx_list]
        mean_point = np.mean(selected_points, axis=0)
        new_points.append(mean_point)

        if new_colors is not None:
            mean_color = np.mean(colors[idx_list], axis=0)
            new_colors.append(mean_color)

        if new_normals is not None:
            # 平均法线需要归一化，以免产生非单位法线
            mean_normal = np.mean(normals[idx_list], axis=0)
            norm_len = np.linalg.norm(mean_normal)
            if norm_len > 1e-12:
                mean_normal = mean_normal / norm_len
            new_normals.append(mean_normal)

    new_points = np.array(new_points, dtype=np.float32)
    if new_colors is not None:
        new_colors = np.array(new_colors, dtype=np.float32)
    if new_normals is not None:
        new_normals = np.array(new_normals, dtype=np.float32)

    downsampled_pc = BasicPointCloud(points=new_points,
                                     colors=new_colors,
                                     normals=new_normals)
    return downsampled_pc
def save_partition_as_pkl(partition, base_directory):
    """
    将单个分区信息保存为 pkl 文件。

    :param partition: 分区对象，包含分区的所有信息。
    :param base_directory: 保存分区文件的基础目录。
    """
    # 创建分区目录
    partition_dir = os.path.join(base_directory, str(partition.partition_id))
    if not os.path.exists(partition_dir):
        os.makedirs(partition_dir)
    # 定义 pkl 文件路径
    pkl_file_path = os.path.join(partition_dir, f"partition_{partition.partition_id}.pkl")
    model_path = os.path.dirname(os.path.dirname(base_directory))
    images_source_path= os.path.join(os.path.dirname(model_path), "images")
    save_partition_data(partition, partition_dir,images_source_path)
    # 保存分区为 pkl 文件
    try:
        with open(pkl_file_path, 'wb') as pkl_file:
            pickle.dump(partition, pkl_file)
        print(f"分区 {partition.partition_id} 已成功保存为 pkl 文件: {pkl_file_path}")
    except Exception as e:
        print(f"保存分区 {partition.partition_id} 失败: {e}")
def visibility_based_camera_selection(partition_list, plot_path):
    """
    根据分区的三维包围盒角点和相机的可见性，选择覆盖目标区域的相机。
    参数:
        partition_list (list): 分区对象列表，每个分区包含点云和相机信息。
        plot_path (str): 用于保存可视化结果的路径。

    返回:
        add_visible_camera_partition_list (list): 更新后的分区列表，包含可见相机和去重后的点云。
    """
    add_visible_camera_partition_list = copy.deepcopy(partition_list)  # 深拷贝分区列表

    for idx, partition_i in enumerate(partition_list):
        new_points = []  # 用于保存新增的点
        new_colors = []
        new_normals = []
        upadata_camera = []  # 用于记录新增的相机
        intersects_partitions = []
        partition_id_i = partition_i.partition_id  # 当前分区编号

        # 查找与当前分区相交的其他分区
        for partition_j in partition_list:
            if partition_i.partition_id == partition_j.partition_id:
                continue
            if partition_i.origin_box.intersects(partition_j.origin_box):
                intersects_partitions.append(partition_j)

        # 对当前分区的点云进行下采样
        downsampled_pc = downsample_point_cloud(partition_i.point_cloud, voxel_size=0.5)

        for partition_j in intersects_partitions:
            partition_id_j = partition_j.partition_id
            if partition_id_i == partition_j.partition_id:
                continue  # 跳过相同分区

            print(f"Now processing partition i:{partition_id_i} and j:{partition_id_j}")

            for cameras_pose in partition_j.camera:
                camera = cameras_pose.camera
                image_points, image_mask = project_points_to_camera(downsampled_pc.points, camera)
                if len(image_points) <= 3:
                    continue  # 点数不足，跳过
                # 使用Graham Scan算法计算可见性比率
                pkg = run_graham_scan(image_points, camera.image_width, camera.image_height)
                visible_rate = 0.25  # 可见性阈值
                if pkg["intersection_rate"] >= visible_rate:
                    # 检查相机是否已存在于当前分区
                    collect_names = [cam.camera.image_name for cam in add_visible_camera_partition_list[idx].camera]
                    if cameras_pose.camera.image_name in collect_names:
                        continue  # 相机已存在，跳过
                    # 添加相机到当前分区
                    add_visible_camera_partition_list[idx].camera.append(cameras_pose)
                    upadata_camera.append(cameras_pose)
                    # 获取可见点云
                    _, mask = project_points_to_camera(partition_j.point_cloud.points, camera)
                    updated_points = partition_j.point_cloud.points[mask]
                    updated_colors = partition_j.point_cloud.colors[mask]
                    updated_normals = partition_j.point_cloud.normals[mask]
                    # 将可见点云添加到新增点云列表
                    new_points.append(updated_points)
                    new_colors.append(updated_colors)
                    new_normals.append(updated_normals)
        # 可视化更新
        plot_single_subregion_with_updates(partition_i, upadata_camera, new_points, plot_path)
        # 获取当前分区的原始点云并添加到新增点云列表
        point_cloud = add_visible_camera_partition_list[idx].point_cloud
        new_points.append(point_cloud.points)
        new_colors.append(point_cloud.colors)
        new_normals.append(point_cloud.normals)
        # 合并所有新增点云
        new_points = np.concatenate(new_points, axis=0)
        new_colors = np.concatenate(new_colors, axis=0)
        new_normals = np.concatenate(new_normals, axis=0)
        # 点云去重优化：使用 Open3D 进行去重
        # 创建 Open3D 点云对象
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(new_points)
        if new_colors.size > 0:
            o3d_pcd.colors = o3d.utility.Vector3dVector(new_colors)
        if new_normals.size > 0:
            o3d_pcd.normals = o3d.utility.Vector3dVector(new_normals)
        # 使用 Open3D 的 voxel_down_sample 算法进行去重
        # 这里 voxel_size 应与下采样时使用的一致
        # 由于已经下采样，voxel_size 可以设置为一个较小的值来进一步去重
        o3d_unique_pcd = o3d_pcd.voxel_down_sample(voxel_size=1e-6)  # 设置一个非常小的 voxel_size 以实现近似去重
        # 转换回 NumPy 数组
        unique_points = np.asarray(o3d_unique_pcd.points)
        unique_colors = np.asarray(o3d_unique_pcd.colors) if o3d_unique_pcd.has_colors() else np.zeros(
            (unique_points.shape[0], 3))
        unique_normals = np.asarray(o3d_unique_pcd.normals) if o3d_unique_pcd.has_normals() else np.zeros(
            (unique_points.shape[0], 3))
        # 更新分区的点云
        add_visible_camera_partition_list[idx] = add_visible_camera_partition_list[idx]._replace(point_cloud=BasicPointCloud(points=unique_points, colors=unique_colors, normals=unique_normals) )
        #保存分区结果
        save_partition_as_pkl(add_visible_camera_partition_list[idx], plot_path)
        # 打印更新信息
        print(f"分区 {idx} 更新后的相机数量: {len(add_visible_camera_partition_list[idx].camera)}")
        print(f"分区 {idx} 更新后的点云数量: {len(unique_points)}")
        print(f"分区 {idx} 更新前的点云数量: {len(partition_i.point_cloud.points)}")
        print("-----------------------------")
    return add_visible_camera_partition_list