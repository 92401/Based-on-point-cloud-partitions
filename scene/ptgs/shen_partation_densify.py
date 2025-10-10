import os
import pickle
import random
import matplotlib.pyplot as plt
import copy
import time
from shapely.geometry.geo import box
from shapely.vectorized import contains
from shapely.strtree import STRtree
from shapely.geometry import Polygon, LineString,Point
from typing import NamedTuple, List, Optional
import math

import numpy as np
import torch
from scene.cameras import SimpleCamera
from scene.ptgs.save_partition import save_partition_data
from shen_data_read import partition, CameraInfo
import open3d as o3d
import density_partition
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
def quad_tree_partition(points, bounds, threshold, depth=0):
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
    partitions=density_partition.density_partition(points,bounds,threshold, depth=0)
    return partitions
def compute_avg_xz_distance(cameras, points, sample_size=10):
    """
    计算每个相机的在 XZ 平面上的平均距离。

    :param cameras: List[Camera]，相机对象列表
    :param points: np.ndarray (N, 3)，所有点云的 xyz 坐标
    :param sample_size: int，每个相机随机采样的点数
    :return: dict，{camera_id: 平均距离}
    """
    continue_num=0
    dictance=0
    camera_distances = {}
    for camera in cameras:
        # 提取相机的坐标 (x, z)
        if isinstance(camera.camera_center, torch.Tensor):
            pose = camera.camera_center.cpu().numpy()
        elif isinstance(camera.camera_center, np.ndarray):
            pose = camera.camera_center
        else:
            pose = np.array(camera.camera_center)
        if camera.points3D_ids is None or camera.points3D_ids.size == 0:
            continue_num+=1
            continue  # 如果该相机没有点，跳过
        valid_ids_mask = camera.points3D_ids != -1
        valid_point3D_ids = camera.points3D_ids[valid_ids_mask]

        # 检查有效的 valid_point3D_ids 是否为空
        if valid_point3D_ids.size == 0:
            continue_num+=1
            continue

        # 过滤掉超出 pcd.points 索引范围的点
        max_index = len(points) - 1
        visible_point_ids = valid_point3D_ids[valid_point3D_ids <= max_index]
        camera_x, camera_z=pose[0], pose[2]  # 提取相机的平移向量 (世界坐标系)


        # 取出相机可见点的坐标
        visible_points = points[visible_point_ids]  # 形状 (M, 3)
        visible_xz = visible_points[:, [0, 2]]  # 只取 x, z 坐标

        # 随机采样部分点（如果点数少于 sample_size，则全部取出）
        sampled_points = visible_xz[random.sample(range(len(visible_xz)), min(sample_size, len(visible_xz)))]

        # 计算这些点到相机的 xz 平面欧几里得距离
        distances = np.linalg.norm(sampled_points - np.array([camera_x, camera_z]), axis=1)

        # 计算平均距离
        avg_distance = np.mean(distances)
        dictance +=avg_distance
        # 存入结果
        camera_distances[camera.colmap_id] = avg_distance  # 假设相机有 camera_id 作为唯一标识符
    a=dictance / (len(cameras) - continue_num)
    return camera_distances,a
def filter_partitions_by_point_count(partitions,points, min_threshold=10):
    partitions = [p for p in partitions if p.point_num >= min_threshold]
    total_points = points.points.shape[0]  # 获取点云的总点数
    lower_threshold = (total_points / len(partitions)) * 0.1 if partitions else 0
    filter_partitions = [p for p in partitions if p.point_num > lower_threshold]
    return filter_partitions
# #消融实验
# def expand_partitions( filtered_partitions: List[Partition], point3D,
#         expansion_distance: float = 0) -> List[Partition]:
#     """
#     根据坐标在 x 和 y 方向上分别扩展每个分区的包围盒，并提取扩展后的点云数据。

#     :param filtered_partitions: 要扩展的分区列表
#     :param point3D: 原始点云数据（Open3D PointCloud 对象）
#     :param expansion_distance: 每个方向上要扩展的距离（默认 100.0）
#     :return: 扩展后的分区列表
#     """

#     import numpy as np
#     from shapely.geometry import Polygon, Point
#     import shapely.vectorized

#     def extract_point_cloud(pcd, polygon):
#         min_x, min_z, max_x, max_z = polygon.bounds
#         bbox=[min_x, max_x, min_z, max_z]
#         """根据camera的边界从初始点云中筛选对应partition的点云"""
#         mask = (pcd.points[:, 0] >= bbox[0]) & (pcd.points[:, 0] <= bbox[1]) & (
#                 pcd.points[:, 1] >= bbox[2]) & (pcd.points[:, 1] <= bbox[3])  # 筛选在范围内的点云，得到对应的mask
#         points = pcd.points[mask]
#         colors = pcd.colors[mask]
#         normals = pcd.normals[mask]
#         return BasicPointCloud(points=points, colors=colors, normals=normals)

#     expanded_partitions = []

#     for partition in filtered_partitions:
#         # 获取原始包围盒的边界
#         minx, miny, maxx, maxy = partition.origin_box.bounds
#         # 计算扩展后的边界
#         new_minx = minx - expansion_distance
#         new_maxx = maxx + expansion_distance
#         new_miny = miny - expansion_distance
#         new_maxy = maxy + expansion_distance
#         width = maxx - minx
#         height = maxy - miny

#         # 扩展比例
#         expand_ratio = 0 # 10%

#         # 计算扩展量
#         expand_width = width * expand_ratio / 2
#         expand_height = height * expand_ratio / 2

#         # 更新边界值
#         point_minx = minx - expand_width
#         point_miny = miny - expand_height
#         point_maxx = maxx + expand_width
#         point_maxy = maxy + expand_height
#         # 创建扩展后的包围盒（Shapely Polygon 对象）
#         expanded_bounds = box(new_minx, new_miny, new_maxx, new_maxy)
#         pointbox=box(point_minx,point_miny,point_maxx,point_maxy)
#         # 提取扩展后的点云数据
#         filtered_basic_pcd = extract_point_cloud(point3D, pointbox)

#         # 创建新的扩展后的分区
#         expanded_partition = Partition(
#             partition_id=partition.partition_id,
#             origin_box=partition.origin_box,
#             camera=[],  # 根据需要填充或保持为空
#             extend_rate=expansion_distance,  # 使用固定扩展距离作为扩展率
#             extend_box=pointbox,
#             point_num=len(filtered_basic_pcd.points),
#             point_cloud=filtered_basic_pcd
#         )
#         expanded_partitions.append(expanded_partition)

#         print( f"分区 {partition.partition_id} 已扩展：原始边界 {partition.origin_box.bounds} -> 扩展后边界 {expanded_bounds.bounds}")
#     return expanded_partitions
def expand_partitions( filtered_partitions: List[Partition], point3D,
        expansion_distance: float = 80) -> List[Partition]:
    """
    根据坐标在 x 和 y 方向上分别扩展每个分区的包围盒，并提取扩展后的点云数据。

    :param filtered_partitions: 要扩展的分区列表
    :param point3D: 原始点云数据（Open3D PointCloud 对象）
    :param expansion_distance: 每个方向上要扩展的距离（默认 100.0）
    :return: 扩展后的分区列表
    """

    import numpy as np
    from shapely.geometry import Polygon, Point
    import shapely.vectorized

    def extract_point_cloud(pcd, polygon):
        min_x, min_z, max_x, max_z = polygon.bounds
        bbox=[min_x, max_x, min_z, max_z]
        """根据camera的边界从初始点云中筛选对应partition的点云"""
        mask = (pcd.points[:, 0] >= bbox[0]) & (pcd.points[:, 0] <= bbox[1]) & (
                pcd.points[:, 1] >= bbox[2]) & (pcd.points[:, 1] <= bbox[3])  # 筛选在范围内的点云，得到对应的mask
        points = pcd.points[mask]
        colors = pcd.colors[mask]
        normals = pcd.normals[mask]
        return BasicPointCloud(points=points, colors=colors, normals=normals)

    expanded_partitions = []

    for partition in filtered_partitions:
        # 获取原始包围盒的边界
        minx, miny, maxx, maxy = partition.origin_box.bounds
        # 计算扩展后的边界
        new_minx = minx - expansion_distance
        new_maxx = maxx + expansion_distance
        new_miny = miny - expansion_distance
        new_maxy = maxy + expansion_distance
        width = maxx - minx
        height = maxy - miny

        # 扩展比例
        expand_ratio = 0.10  # 10%

        # 计算扩展量
        expand_width = width * expand_ratio / 2
        expand_height = height * expand_ratio / 2

        # 更新边界值
        point_minx = minx - expand_width
        point_miny = miny - expand_height
        point_maxx = maxx + expand_width
        point_maxy = maxy + expand_height
        # 创建扩展后的包围盒（Shapely Polygon 对象）
        expanded_bounds = box(new_minx, new_miny, new_maxx, new_maxy)
        pointbox=box(point_minx,point_miny,point_maxx,point_maxy)
        # 提取扩展后的点云数据
        filtered_basic_pcd = extract_point_cloud(point3D, pointbox)

        # 创建新的扩展后的分区
        expanded_partition = Partition(
            partition_id=partition.partition_id,
            origin_box=partition.origin_box,
            camera=[],  # 根据需要填充或保持为空
            extend_rate=expansion_distance,  # 使用固定扩展距离作为扩展率
            extend_box=pointbox,
            point_num=len(filtered_basic_pcd.points),
            point_cloud=filtered_basic_pcd
        )
        expanded_partitions.append(expanded_partition)

        print( f"分区 {partition.partition_id} 已扩展：原始边界 {partition.origin_box.bounds} -> 扩展后边界 {expanded_bounds.bounds}")
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

def plot_quad_tree_extend(partitions, points, picname):  #适用于不加入相机的画图
    fig, ax = plt.subplots(figsize=(12, 8))  # 调整图形尺寸
    # 绘制点云
    xs, ys = zip(*points)
    ax.scatter(xs, ys, s=10, c="blue", alpha=0.5)  # 调整点的大小
    # 为每个区域分配一个随机颜色
    colors = [tuple(random.random() for _ in range(3)) for _ in range(len(partitions))]
    # 绘制分区
    for partition in partitions:
        idx = partition.partition_id
        bounds = partition.extend_box
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

def assign_cameras_to_partitions(expanded_partitions: List[Partition], train_cameras: List[SimpleCamera]):
    """
    读取相机信息，将相机加入到各个分区中。

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
            # 检查相机是否在当前分区的原始边界内
            if partition.extend_box.contains(Point((camera_pose.pose[0], camera_pose.pose[1]))):  #xz平面
                cameras_in_partition.append(camera_pose)
            # if partition.extend_box.contains(Point(camera_pose.pose[:2])):  #xy平面
            #     cameras_in_partition.append(camera_pose)
        # 创建新的 Partition 实例，包含相机信息
        updated_partition = Partition(
            partition_id=partition.partition_id,
            camera=cameras_in_partition,  # 存储的相机pose列表
            point_cloud=partition.point_cloud,
            origin_box=partition.origin_box,
            point_num=partition.point_num,
            extend_rate=partition.extend_rate,
            extend_box=partition.extend_box
        )
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
    R_wc = torch.tensor(camera.R, dtype=torch.float32, device=device)
    T_wc = torch.tensor(camera.T, dtype=torch.float32, device=device)
    W2C = torch.zeros((4, 4), dtype=torch.float32, device=device)  # 在GPU上
    W2C[:3, :3] = R_wc
    W2C[:3, -1] = T_wc
    W2C[3, 3] = 1.0
    # 将点云转换为齐次坐标 (N,4)
    N = points_t.shape[0]
    ones = torch.ones((N, 1), dtype=torch.float32, device=device)
    points_homog = torch.cat((points_t, ones), dim=1)  # (N,4)
    # 世界坐标 -> 相机坐标
    points_camera_homog = (W2C @ points_homog.transpose(0, 1)).transpose(0, 1)  # (N,4)
    points_camera = points_camera_homog[:, :3] / points_camera_homog[:, 3, None]  # (N,3)
    # 过滤掉相机后方的点(Z>0)
    in_front_mask = points_camera[:, 1] > 0    #要保证相机镜头朝z轴正的方向
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

    # points_image_homog = (intrinsic_matrix @ points_camera.transpose(0, 1)).transpose(0, 1)
    points_image_homog = (intrinsic_matrix @ points_camera_in_front.transpose(0, 1)).transpose(0, 1)   #不加过滤
    points_image = points_image_homog[:, :2] / points_image_homog[:, 2, None]
    # 检查点是否在图像范围内
    in_image = (points_image[:, 0] >= 0) & (points_image[:, 0] < camera.image_width) & \
               (points_image[:, 1] >= 0) & (points_image[:, 1] < camera.image_height)
    valid_mask = torch.zeros(N, dtype=torch.bool, device=device)
    valid_mask[in_front_mask] = in_image
    # 将结果转回CPU
    projected_points_list = points_image[in_image].cpu().numpy().tolist()
    # valid_mask = valid_mask.cpu().numpy()
    return projected_points_list



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
def process_camera(camera, pcd):
    # 检查 points3D_ids 是否为空
    if camera.points3D_ids is None or camera.points3D_ids.size == 0:
        # print(f"Skipping camera with empty points3D_ids: {camera.image_name}")
        return None, None, None

    # 有效的 points3D_ids（过滤掉 -1 的无效点）
    valid_ids_mask = camera.points3D_ids != -1
    valid_point3D_ids = camera.points3D_ids[valid_ids_mask]

    # 检查有效的 valid_point3D_ids 是否为空
    if valid_point3D_ids.size == 0:
        print(f"No valid points for camera: {camera.image_name}")
        return None, None, None

    # 过滤掉超出 pcd.points 索引范围的点
    max_index = len(pcd.points) - 1
    valid_point3D_ids = valid_point3D_ids[valid_point3D_ids <= max_index]

    # 如果过滤后没有有效的点
    if valid_point3D_ids.size == 0:
        print(f"All points for camera {camera.image_name} are out of bounds.")
        return None, None, None

    # 使用过滤后的 valid_point3D_ids 从点云中提取对应数据
    updated_points = pcd.points[valid_point3D_ids]
    updated_colors = pcd.colors[valid_point3D_ids]
    updated_normals = pcd.normals[valid_point3D_ids]


    return updated_points, updated_colors, updated_normals


# def visibility_based_camera_selection(
#         partition_list: List[Partition],
#         plot_path: str,
#         pcd: BasicPointCloud
# ) -> List[Partition]:
#     """
#     根据分区已分配的相机可见性，选择覆盖目标区域的相机。
#
#     参数:
#         partition_list (List[Partition]): 分区对象列表，每个分区包含点云和相机信息。
#         plot_path (str): 用于保存可视化结果的路径。
#         pcd (BasicPointCloud): 原始点云数据。
#     返回:
#         List[Partition]: 更新后的分区列表，包含可见相机和去重后的点云。
#     """
#     # 深拷贝分区列表以避免修改原始数据
#     updated_partitions = copy.deepcopy(partition_list)
#
#     for idx, partition in enumerate(partition_list):
#         new_points = []  # 用于保存新增的点
#         new_colors = []
#         new_normals = []
#         removed_cameras = []  # 用于记录被剔除的相机
#         added_cameras = []  # 用于记录新增的相机
#         print(f"处理分区 {partition.partition_id} ...")
#         pcd_i = downsample_point_cloud(partition.point_cloud,voxel_size=0.1)
#         # 遍历当前分区已分配的相机
#         for camera_pose in partition.camera:
#             camera = camera_pose.camera
#             try:
#                 # 可见性分析：将分区的点云投影到相机视图中
#                 image_points = project_points_to_camera(pcd_i.points, camera)
#                 if len(image_points) <= 3:
#                     # 点数不足，移除相机
#                     removed_cameras.append(camera_pose)
#                     continue  # 跳过
#                 # 使用 Graham Scan 算法计算可见性比率
#                 pkg = run_graham_scan(image_points, camera.image_width, camera.image_height)
#
#                 visible_rate_threshold = 0.3  # 可见性阈值
#                 if pkg["intersection_rate"] >= visible_rate_threshold:
#                     # 相机满足可见性，保留并添加对应点云
#                     added_cameras.append(camera_pose)
#                     # 获取可见点云
#                     # print('提取可视相机点云')
#                     visible_points,visible_normals,visible_colors = process_camera(camera, pcd)  # 使用分区的点云
#                     if visible_points is None:
#                         # print('该相机不包含点云')
#                         continue
#                     # 合并可见点云
#                     print('点云合并')
#                     new_points.append(visible_points)
#                     new_colors.append(visible_colors)
#                     new_normals.append(visible_normals)
#                     # end = time.time()
#                     # print('添加相机时间', end - star)
#                 else:
#                     # 相机不满足可见性，移除相机
#                     print('可见性比率小移除相机')
#                     removed_cameras.append(camera_pose)
#             except Exception as e:
#                 print(f"处理相机 {camera.image_name} 时发生错误: {e}")
#                 removed_cameras.append(camera_pose)
#                 continue
#
#         # 更新相机列表：只保留满足可见性的相机
#         # star1=time.time()
#         updated_partitions[idx] = updated_partitions[idx]._replace(camera=added_cameras)
#         # 获取当前分区的原始点云并添加到新增点云列表
#         current_pcd = updated_partitions[idx].point_cloud
#         new_points.append(current_pcd.points)
#         new_colors.append(current_pcd.colors)
#         new_normals.append(current_pcd.normals)
#         # 合并所有新增点云
#         new_points = np.concatenate(new_points, axis=0)
#         new_colors = np.concatenate(new_colors, axis=0) if new_colors else None
#         new_normals = np.concatenate(new_normals, axis=0) if new_normals else None
#         o3d_pcd = o3d.geometry.PointCloud()
#         o3d_pcd.points = o3d.utility.Vector3dVector(new_points)
#         o3d_pcd.colors = o3d.utility.Vector3dVector(new_colors)
#         o3d_pcd.normals = o3d.utility.Vector3dVector(new_normals)
#         # 使用 Open3D 的 voxel_down_sample 算法进行去重
#         # 这里 voxel_size 应与下采样时使用的一致
#         # 由于已经下采样，voxel_size 可以设置为一个较小的值来进一步去重
#         o3d_unique_pcd = o3d_pcd.voxel_down_sample(voxel_size=1e-6)  # 设置一个非常小的 voxel_size 以实现近似去重
#         # 转换回 NumPy 数组
#         unique_points = np.asarray(o3d_unique_pcd.points)
#         unique_colors = np.asarray(o3d_unique_pcd.colors) if o3d_unique_pcd.has_colors() else np.zeros(
#             (unique_points.shape[0], 3))
#         unique_normals = np.asarray(o3d_unique_pcd.normals) if o3d_unique_pcd.has_normals() else np.zeros(
#             (unique_points.shape[0], 3))
#         # 更新分区的点云
#         updated_partitions[idx] = updated_partitions[idx]._replace(
#             point_cloud=BasicPointCloud(points=unique_points, colors=unique_colors, normals=unique_normals))
#         save_partition_as_pkl(updated_partitions[idx], plot_path)
#         # 打印更新信息
#         print(f"分区 {idx} 更新后的相机数量: {len(updated_partitions[idx].camera)}")
#         print(f"分区 {idx} 更新后的点云数量: {len(unique_points)}")
#         print(f"分区 {idx} 更新前的点云数量: {len(partition.point_cloud.points)}")
#         print("-----------------------------")
#     return updated_partitions
import concurrent.futures
import copy
from typing import List


# def process_camera_visibility(camera_pose, pcd_i, visible_rate_threshold, pcd):
#     """
#     处理单个相机的可见性分析，并返回其可见点云和更新状态
#     """
#     camera = camera_pose.camera
#     try:
#         # 可见性分析：将分区的点云投影到相机视图中
#         image_points = project_points_to_camera(pcd_i.points, camera)
#         if len(image_points) <= 3:
#             return None, None  # 该相机不满足可见性要求

#         # 使用 Graham Scan 计算可见性比率
#         pkg = run_graham_scan(image_points, camera.image_width, camera.image_height)

#         if pkg["intersection_rate"] >= visible_rate_threshold:
#             # 相机满足可见性要求，获取可见点云
#             visible_points, visible_normals, visible_colors = process_camera(camera, pcd)
#             if visible_points is not None:
#                 return camera_pose, (visible_points, visible_normals, visible_colors)

#     except Exception as e:
#         print(f"处理相机 {camera.image_name} 时发生错误: {e}")

#     return None, None  # 该相机未被选择


# def visibility_based_camera_selection(
#         partition_list: List[Partition],
#         plot_path: str,
#         pcd: BasicPointCloud,
#         max_workers: int = 48
# ) -> List[Partition]:
#     """
#     使用多线程处理相机可见性分析，提高处理速度
#     """
#     updated_partitions = copy.deepcopy(partition_list)

#     for idx, partition in enumerate(partition_list):
#         new_points = []  # 用于保存新增的点
#         new_colors = []
#         new_normals = []
#         added_cameras = []  # 记录保留的相机

#         print(f"处理分区 {partition.partition_id} ...")

#         # 对点云进行下采样，加速计算
#         pcd_i = downsample_point_cloud(partition.point_cloud, voxel_size=0.1)
#         print('相机投影中...')
#         # 并行处理所有相机的可见性分析
#         visible_rate_threshold = 0.3  # 可见性阈值
#         with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#             futures = {
#                 executor.submit(process_camera_visibility, camera_pose, pcd_i, visible_rate_threshold, pcd): camera_pose
#                 for camera_pose in partition.camera
#             }

#             for future in concurrent.futures.as_completed(futures):
#                 result = future.result()
#                 if result[0] is not None:
#                     added_cameras.append(result[0])  # 相机
#                     new_points.append(result[1][0])  # 可见点云
#                     new_colors.append(result[1][1])  # 颜色
#                     new_normals.append(result[1][2])  # 法线

#         # 更新相机列表
#         updated_partitions[idx] = updated_partitions[idx]._replace(camera=added_cameras)

#         # 追加原始点云
#         current_pcd = updated_partitions[idx].point_cloud
#         new_points.append(current_pcd.points)
#         new_colors.append(current_pcd.colors)
#         new_normals.append(current_pcd.normals)

#         # 合并点云数据
#         new_points = np.concatenate(new_points, axis=0)
#         new_colors = np.concatenate(new_colors, axis=0) if new_colors else None
#         new_normals = np.concatenate(new_normals, axis=0) if new_normals else None

#         # # 创建 Open3D 点云对象
#         # o3d_pcd = o3d.geometry.PointCloud()
#         # o3d_pcd.points = o3d.utility.Vector3dVector(new_points)
#         # o3d_pcd.colors = o3d.utility.Vector3dVector(new_colors)
#         # o3d_pcd.normals = o3d.utility.Vector3dVector(new_normals)
#         #
#         # # 进一步去重
#         # o3d_unique_pcd = o3d_pcd.voxel_down_sample(voxel_size=1e-6)
#         #
#         # # 转换回 NumPy
#         # unique_points = np.asarray(o3d_unique_pcd.points)
#         # unique_colors = np.asarray(o3d_unique_pcd.colors) if o3d_unique_pcd.has_colors() else np.zeros(
#         #     (unique_points.shape[0], 3))
#         # unique_normals = np.asarray(o3d_unique_pcd.normals) if o3d_unique_pcd.has_normals() else np.zeros(
#         #     (unique_points.shape[0], 3))
#         new_points, mask = np.unique(new_points, return_index=True, axis=0)
#         new_colors = new_colors[mask]
#         new_normals = new_normals[mask]
#         # 更新分区
#         updated_partitions[idx] = updated_partitions[idx]._replace(
#             point_cloud=BasicPointCloud(points=new_points, colors=new_colors, normals=new_normals))
#         save_partition_as_pkl(updated_partitions[idx], plot_path)

#         # 打印更新信息
#         print(f"分区 {idx} 更新后的相机数量: {len(updated_partitions[idx].camera)}")
#         print(f"分区 {idx} 更新后的点云数量: {len(new_points)}")
#         print(f"分区 {idx} 更新前的点云数量: {len(partition.point_cloud.points)}")
#         print("-----------------------------")

#     return updated_partitions


# #消融实验
def process_camera_visibility(camera_pose, pcd_i, visible_rate_threshold, pcd):
    """
    处理单个相机的可见性分析，只返回满足可见性的相机，不再返回点云
    """
    camera = camera_pose.camera
    try:
        # 将点投影到相机视图
        image_points = project_points_to_camera(pcd_i.points, camera)
        if len(image_points) <= 3:
            return None

        # 使用 Graham Scan 计算可见区域比例
        pkg = run_graham_scan(image_points, camera.image_width, camera.image_height)

        if pkg["intersection_rate"] >= visible_rate_threshold:
            return camera_pose  # 满足可见性，返回相机

    except Exception as e:
        print(f"处理相机 {camera.image_name} 时发生错误: {e}")

    return None  # 不满足可见性
def visibility_based_camera_selection(
        partition_list: List[Partition],
        plot_path: str,
        pcd: BasicPointCloud,
        max_workers: int = 48
) -> List[Partition]:
    """
    基于可见性筛选相机，不修改原有点云，只更新相机列表
    """
    updated_partitions = copy.deepcopy(partition_list)

    for idx, partition in enumerate(partition_list):
        added_cameras = []  # 记录保留的相机

        print(f"处理分区 {partition.partition_id} ...")

        # 对点云进行下采样，加速可见性分析
        pcd_i = downsample_point_cloud(partition.point_cloud, voxel_size=0.1)

        visible_rate_threshold = 0.45  # 可见性阈值
        print('相机投影中...')
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_camera_visibility, camera_pose, pcd_i, visible_rate_threshold, pcd): camera_pose
                for camera_pose in partition.camera
            }

            for future in concurrent.futures.as_completed(futures):
                camera_pose_result = future.result()
                if camera_pose_result is not None:
                    added_cameras.append(camera_pose_result)

        # 更新分区中的相机列表
        updated_partitions[idx] = updated_partitions[idx]._replace(camera=added_cameras)

        # 保存更新后的分区
        save_partition_as_pkl(updated_partitions[idx], plot_path)

        # 打印更新信息
        print(f"分区 {idx} 更新后的相机数量: {len(updated_partitions[idx].camera)}")
        print(f"分区 {idx} 点云数量（未修改）: {len(partition.point_cloud.points)}")
        print("-----------------------------")

    return updated_partitions
