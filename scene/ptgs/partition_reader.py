#场景信息和相机信息
import os
import sys
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
from plyfile import PlyData, PlyElement
from scene.gaussian_model import BasicPointCloud

#文件的功能是读取三个bin文件返回一个场景信息，场景包括点云对象，由camerainfo组合的测试和训练列表，xyz_rgb的稀疏点云存储路径

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
def readColmapSceneInfoVast(path, model_path, partition_id, images, eval, man_trans, llffhold=83):
    # 读取每个partition的点云，以及对应的相机
    # 读取所有图像的信息，包括相机内外参数，以及3D点云坐标
    client_camera_txt_path = os.path.join(model_path, f"{partition_id}_camera.txt")
    with open(client_camera_txt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]

    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary_vast(cameras_extrinsic_file, lines)
    cam_intrinsics = read_intrinsics_binary_vast(cameras_intrinsic_file, lines)

    images_dir = os.path.join(path, "images")
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=images_dir,
                                           man_trans=man_trans)  # 存储所有图片的 相机模型id，旋转矩阵 平移向量，视角场，图片数据，图片路径，图片名，图片宽高
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)  # 根据图片名称对 list进行排序

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos  # 得到训练图片的相机参数
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)  # 使用找到在世界坐标系下相机的几何中心

    ply_path = os.path.join(model_path, f"{partition_id}_visible.ply")
    pcd = fetchPly(ply_path, man_trans=None)  # 得到稀疏点云中，各个3D点的属性信息，点云已经经过曼哈顿对齐，第二次加载不需要再进行对齐，否则点云坐标会发生变换
    # print(pcd)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)  # 保存一个场景的所有参数信息
    return scene_info

def readColmapCamerasPartition(cam_extrinsics, cam_intrinsics, images_folder, man_trans):
    """
    根据分区的三维包围盒角点和相机的可见性，选择覆盖目标区域的相机。

    :param cam_extrinsics: 字典，包含相机外参，每个键对应一个相机的外参信息（包含 qvec 和 tvec）
    :param cam_intrinsics: 字典，包含相机内参，每个键对应一个相机的内参信息（包含 id, height, width, model, params）
    :param images_folder: 字符串，图片文件夹的路径
    :param man_trans: 4x4 手动变换矩阵（numpy数组），用于调整相机坐标系，若无则为 None
    :return: cam_infos，包含所有相机信息的列表
    """
    cam_infos = []
    total_cameras = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):  # 每个相机单独处理
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, total_cameras))
        sys.stdout.flush()

        extr = cam_extrinsics[key]  # 获取该相机的外参
        intr = cam_intrinsics[extr.camera_id]  # 获取该相机的内参
        height = intr.height  # 获取图片高度
        width = intr.width  # 获取图片宽度
        uid = intr.id  # 获取相机对应的ID
        # 从四元数获取世界到相机的旋转矩阵
        R_wc = qvec2rotmat(extr.qvec)  # qvec2rotmat 应返回世界到相机的旋转矩阵 (3x3)
        T_wc = np.array(extr.tvec).reshape(3, 1)  # 获取世界到相机的平移向量 (3x1)

        if man_trans is not None:
            # 构建世界到相机的齐次变换矩阵
            W2C = np.eye(4)
            W2C[:3, :3] = R_wc
            W2C[:3, 3] = T_wc.flatten()
            # 应用手动变换：新的世界到相机变换
            W2nC = W2C @ np.linalg.inv(man_trans)
            # 提取新的旋转矩阵和平移向量
            R_wc = W2nC[:3, :3]
            T_wc = W2nC[:3, 3].reshape(3, 1)
        # 处理相机内参
        params = np.array(intr.params)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]  # 简单针孔模型只有一个焦距参数
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]  # 针孔模型有两个焦距参数
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            raise ValueError(
                "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!")

        # 构建图片路径和名称
        image_path = os.path.join(images_folder, os.path.basename(extr.name))  # 获取该图片路径
        image_name = os.path.basename(image_path).split(".")[0]  # 获取该图片名称
        image = None  # 此处不加载图片

        # 创建 CameraInfo 对象（假设 CameraInfo 已定义）
        cam_info = CameraInfo(uid=uid, R=R_wc, T=T_wc, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)

        cam_infos.append(cam_info)  # 存储所有相机的信息

    sys.stdout.write('\n')
    return cam_infos

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):  #列表中的所有相机中心点按列堆叠成一个二维数组。
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate, "radius": radius}


def fetchPly(path, man_trans=None):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']  # 提取点云的顶点
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T  # 将x,y,z这三个坐标属性堆叠在一起
    # print(positions.shape)
    if man_trans is not None:  # 曼哈顿对齐
        man_trans_R = man_trans[:3, :3]
        man_trans_T = man_trans[:3, -1]
        new_positions = np.dot(man_trans_R, positions.transpose()) + np.repeat(man_trans_T, positions.shape[0]).reshape(
            -1, positions.shape[0])
        positions = new_positions.transpose()
    colors = np.vstack(
        [vertices['red'], vertices['green'], vertices['blue']]).T / 255.0  # 将R,G,B三个颜色属性堆叠在一起，并除以255进行归一化
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T  # 提取顶点的三个法向量属性，并堆叠在一起
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def partition(path, images, man_trans, eval=False, llffhold=83):
    # 读取整个场景的点云和相机参数，用于分块
    # 读取所有图像的信息，包括相机内外参数，以及3D点云坐标
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")  # 相机外参文件
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")  # 相机内参文件
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)  # 读取相机外参
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)  # 读取相机内参
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    #image
    reading_dir = os.path.join(path, "images")
    cam_infos_unsorted = readColmapCamerasPartition(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                                    images_folder=reading_dir,
                                                    man_trans=man_trans)  # 存储所有图片的 相机模型id，旋转矩阵 平移向量，视角场，图片数据，图片路径，图片名，图片宽高
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)  # 根据图片名称对 list进行排序

    #根据llffhold分块排序好的相机列表，将相机信息分成训练集和测试集
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)  # 使用找到在训练世界坐标系下相机的几何中心
    # 将3D点云数据写入 scene_info中
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene.")  # 将point3d.bin转换为.ply，只会在您第一次打开场景时发生。
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)  #自己脚本保存ply的部分
    pcd = fetchPly(ply_path, man_trans=man_trans)  # 得到稀疏点云中，各个3D点的属性信息（坐标颜色法线），有曼哈顿的进行旋转

    #这部分不一定会使用
    dist_threshold = 99
    points, colors, normals = pcd.points, pcd.colors, pcd.normals
    points_threshold = np.percentile(points[:, 1], dist_threshold)  # use dist_ratio to exclude outliers
    colors = colors[points[:, 1] < points_threshold]
    normals = normals[points[:, 1] < points_threshold]
    points = points[points[:, 1] < points_threshold]
    pcd = BasicPointCloud(points=points, colors=colors, normals=normals)  #去除离群点

    # print(pcd)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,   #世界坐标系下的相机几何中心
                           ply_path=ply_path)  # 保存一个场景的所有参数信息
    return scene_info
