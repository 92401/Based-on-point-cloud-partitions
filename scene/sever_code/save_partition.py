import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R
from scene.ptgs import read_write_model
from scene.ptgs.read_write_model import Image
from scene.ptgs.shen_data_read import storePly

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
    # read_write_model.write_points3D_binary(partition.point_cloud, os.path.join(partition_dir, "points3D.bin"))
    storePly(os.path.join(sparse_dir, "points3D.ply"),partition.point_cloud.points,partition.point_cloud.colors)
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
        print(f"图片文件名: {image_name}")  # 打印文件名进行检查
        # 构建完整的源路径，假设图片是 .jpg 格式
        source_path = os.path.join(images_path, image_name + ".jpg")
        print(f"构建的源路径: {source_path}")  # 打印路径进行检查
        # 构建目标图片路径
        target_path = os.path.join(target_dir, os.path.basename(image_name) + ".jpg")
        # 判断源文件是否存在，避免文件不存在时发生错误
        if os.path.exists(source_path):
            shutil.copy2(source_path,target_path)  # 使用 copy2 保留文件的元数据（如修改时间等）
            print(f"图片 {image_name} 已成功复制到 {target_path}")
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




