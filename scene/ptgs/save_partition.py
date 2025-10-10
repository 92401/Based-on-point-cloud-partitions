import os
import shutil
import numpy as np
import torch
from typing import NamedTuple
from scipy.spatial.transform import Rotation as R
from scene.ptgs import read_write_model
from shen_data_read import CameraInfo
from scene.ptgs.read_write_model import Image
from scene.ptgs.shen_data_read import storePly

class CameraPose(NamedTuple):
    camera: CameraInfo
    pose: np.array  # [x, y, z] 坐标
class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
def save_partition_data(partition, base_dir: str,imgaes_source_path):
    """
    为每个分区保存数据到单独的文件夹中。
    :param partition: 分区对象，包含相机信息和点云数据
    :param base_dir: 基础目录路径
    imgaes_source_path存储大场景所有图片的路径
    """
    partition_dir = os.path.join(base_dir, f"partition_{partition.partition_id}")
    os.makedirs(partition_dir, exist_ok=True)
    # 创建 images 文件夹
    images_dir = os.path.join(partition_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    sparse_dir = os.path.join(partition_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    # 保存相机信息到 images.bin
    images=simple_camera_to_images(partition.camera)
    read_write_model.write_images_binary(images, os.path.join(sparse_dir, "images.bin"))
    # 保存点云数据到 points3D.ply
    points3D = convert_to_colmap_format(partition.point_cloud) #转换为colmap点云类
    read_write_model.write_points3D_binary(points3D, os.path.join(sparse_dir, "points3D.bin"))
    # storePly(os.path.join(sparse_dir, "points3D.ply"),partition.point_cloud.points,partition.point_cloud.colors)
    # 复制图片到 images 文件夹
    copy_images(partition.camera,imgaes_source_path, images_dir)
    project_root = os.path.dirname(imgaes_source_path)
    # 构建 source_cameras_path
    source_cameras_path = os.path.join(project_root, "sparse", "0", "cameras.bin")
    # 构建 base_partition_path
    base_partition_path = os.path.join(project_root, "model", "split_result", "visible")
    # 复制 cameras.bin 文件到所有子分区的文件夹
    copy_cameras_to_partitions(source_cameras_path, base_partition_path)

def copy_cameras_to_partitions(source_path, base_partition_path):
    """
    将 cameras.bin 文件复制到所有子分区的文件夹中。
    :param source_path: cameras.bin 文件的源路径
    :param base_partition_path: 分区文件夹的基础路径
    """
    # 确保源文件存在
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")
    # 获取所有分区文件夹
    partition_folders = [f for f in os.listdir(base_partition_path) if os.path.isdir(os.path.join(base_partition_path, f))]
    for folder in partition_folders:
        partition_dir = os.path.join(base_partition_path, folder, f"partition_{folder}")
        # 确保目标文件夹存在
        os.makedirs(partition_dir, exist_ok=True)
        # 复制 cameras.bin 文件到目标路径
        target_path = os.path.join(partition_dir, "sparse", "0",'cameras.bin')
        try:
            shutil.copy2(source_path, target_path)
            print(f"Copied cameras.bin to {target_path}")
        except Exception as e:
            print(f"Error copying file: {e}")

def copy_images(cameras,image_file,target_dir: str):
    """
    复制相机对应的图片到目标文件夹。
    :param cameras: 相机信息列表
    :param target_dir: 目标文件夹路径
    """
    images_path = image_file  # 原始图片的存储路径
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for camera_pose in cameras:
        camera = camera_pose.camera
        # 获取相机的 image_name（假设它不带后缀）
        image_name = camera.image_name  # 获取图片的文件名（不带后缀）
        # print(f"图片文件名: {image_name}")  # 打印文件名进行检查
        # 构建完整的源路径，假设图片是 .jpg 格式
        source_path = os.path.join(images_path, image_name + ".jpg")
        # print(f"构建的源路径: {source_path}")  # 打印路径进行检查
        # 构建目标图片路径
        target_path = os.path.join(target_dir, os.path.basename(image_name) + ".jpg")
        # 判断源文件是否存在，避免文件不存在时发生错误
        if os.path.exists(source_path):
            shutil.copy2(source_path,target_path)  # 使用 copy2 保留文件的元数据（如修改时间等）
            # print(f"图片 {image_name} 已成功复制到 {target_path}")
        else:
            print(f"警告: 图片 {image_name} 在源路径 {source_path} 不存在！")

def simple_camera_to_images(cameras):
    """
    将 SimpleCamera 的列表转换为符合 COLMAP 格式的 images 字典。

    Args:
        cameras (list[SimpleCamera]): camera_pose的列表。
    Returns:
        dict: 符合 COLMAP 格式的 images 字典。
    """
    images = {}
    for camera_pose in cameras:  # camera_pose为CameraPose
        camera = camera_pose.camera
        # 将旋转矩阵转换为四元数
        qvec = R.from_matrix(camera.R).as_quat()  # 顺序是 [qx, qy, qz, qw]
        qvec = [qvec[3], qvec[0], qvec[1], qvec[2]]  # 调整为 [qw, qx, qy, qz]

        # 假设 camera.T 是 numpy array 或者可通过 np.array() 转换成numpy array
        tvec_array = np.array(camera.T)
        tvec_list = tvec_array.reshape(-1).tolist()  # 展平成一维列表，再tolist()

        # 创建 Image 实例
        image = Image(
            id=camera.uid,
            qvec=qvec,
            tvec=tvec_list,
            camera_id=camera.colmap_id,
            name=camera.image_name + '.jpg',
            xys=[],  # 如果没有xys信息，可以保持为空列表
            point3D_ids=[]  # 没有3D点ID信息
        )
        images[camera.uid] = image
    return images

import numpy as np
from collections import namedtuple

# 定义目标类型（与函数要求一致）
Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

def convert_to_colmap_format(basic_cloud: BasicPointCloud) -> dict:
    """将 BasicPointCloud 转换为 COLMAP 二进制格式需要的字典"""
    points3D_dict = {}
    
    # 颜色值归一化到 [0, 255]（如果原始范围是 [0,1]）
    colors = basic_cloud.colors
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    
    # 为每个点生成 Point3D 对象并存入字典
    for idx in range(len(basic_cloud.points)):
        points3D_dict[idx] = Point3D(
            id=idx,
            xyz=np.array(basic_cloud.points[idx], dtype=np.float64),  # 必须转为np.array
            rgb=np.array(colors[idx], dtype=np.uint8),                # 必须转为np.array
            error=0.0,                                               # 默认误差
            image_ids=np.array([], dtype=np.int32),                  # 空数组（无关联图像）
            point2D_idxs=np.array([], dtype=np.int32)               # 空数组
        )
    return points3D_dict

def save_test_cameras1(camera_list,imgaes_source_path):
    images = simple_camera_to_images(camera_list)
    sparse_dir = os.path.join(imgaes_source_path, "test","sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    read_write_model.write_images_binary(images, os.path.join(sparse_dir, "images.bin"))
    source_cameras_path = os.path.join(imgaes_source_path, "sparse", "0", "cameras.bin")
    # 复制camera.bin到sparse文件夹下
    shutil.copy2(source_cameras_path, os.path.join(sparse_dir, "cameras.bin"))


def save_test_cameras(camera_list, images_source_path):
    """
    复制 cameras.bin 到 test/sparse/0 并保存 images.bin
    """
    CameraPose_list = []
    camera_centers = []
    for idx, camera in enumerate(camera_list):
        # 确保 camera_center 是 CPU tensor 或 NumPy 数组
        if isinstance(camera.camera_center, torch.Tensor):
            pose = camera.camera_center.cpu().numpy()
        elif isinstance(camera.camera_center, np.ndarray):
            pose = camera.camera_center
        else:
            pose = np.array(camera.camera_center)
        camera_centers.append(pose)
        CameraPose_list.append(CameraPose(camera=camera, pose=pose))
    # 生成 images.bin
    images = simple_camera_to_images(CameraPose_list)
    print(images_source_path)
    # 创建 test/sparse/0 目录
    test_path=os.path.join(images_source_path, "test")
    test_image=os.path.join(test_path,'images')
    sparse_dir = os.path.join(test_path, "sparse", "0")
    images_path = os.path.join(images_source_path, 'images')
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(images_path,exist_ok=True)
    copy_images(CameraPose_list,images_path,test_image)
    # 写入 images.bin
    read_write_model.write_images_binary(images, os.path.join(sparse_dir, "images.bin"))

    # 复制 cameras.bin
    source_cameras_path = os.path.join(images_source_path, "sparse", "0", "cameras.bin")

    if not os.path.exists(source_cameras_path):
        raise FileNotFoundError(f"未找到 cameras.bin 文件: {source_cameras_path}")

    shutil.copy(source_cameras_path, os.path.join(sparse_dir, "cameras.bin"))  # 复制 cameras.bin

    print(f"✅ cameras.bin 复制完成 -> {sparse_dir}")