import os.path
import pickle
import numpy as np
import torch
from plyfile import PlyData

from shapely.vectorized import contains
from sklearn.neighbors import KDTree

from scene.gaussian_model import GaussianModel
def load_ply(path):
    """
    从点云文件加载点和相关属性
    """
    plydata = PlyData.read(path)
    max_sh_degree = 3
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return xyz, features_dc, features_extra, opacities, scales, rots


def extract_point_cloud(xyz, min_neighbors=10, radius=0.1):
    """
    根据三维点云数据去除离散点。
    :param xyz: numpy数组，形状为(N, 3)，表示点云的x, y, z坐标
    :param min_neighbors: 邻域中最少需要的点数（用于判断点是否为离散点）
    :param radius: 邻域搜索半径
    :return: 布尔掩码数组，表示每个点是否为有效点
    """
    # 使用KDTree计算点的邻域密度
    tree = KDTree(xyz)
    neighbors_count = tree.query_radius(xyz, r=radius, count_only=True)

    # 创建布尔掩码，保留满足邻域点数大于等于 min_neighbors 的点
    mask = neighbors_count >= min_neighbors
    return mask
def cut_partition(model_path, partition):
    """
    裁剪分区点云并保存
    """
    save_merge_dir = os.path.join(model_path, "cutor_point_cloud.ply")
    oribox = partition.origin_box
    point_cloud_path = os.path.join(model_path, "point_cloud.ply")
    xyz, features_dc, features_extra, opacities, scales, rots = load_ply(point_cloud_path)
    mask = extract_point_cloud(xyz)
    points = np.array(xyz[mask])
    features_dc_list = np.array(features_dc[mask])
    features_extra_list = np.array(features_extra[mask])
    opacities_list = np.array(opacities[mask])
    scales_list = np.array(scales[mask])
    rots_list = np.array(rots[mask])

    global_model = GaussianModel(3)
    global_params = {'xyz': torch.from_numpy(points).float().cuda(),
                     'rotation': torch.from_numpy(rots_list).float().cuda(),
                     'scaling': torch.from_numpy(scales_list).float().cuda(),
                     'opacity': torch.from_numpy(opacities_list).float().cuda(),
                     'features_dc': torch.from_numpy(features_dc_list).float().cuda().permute(0, 2, 1),
                     'features_rest': torch.from_numpy(features_extra_list).float().cuda().permute(0, 2, 1)}
    global_model.set_params(global_params)
    global_model.save_ply(save_merge_dir)

if __name__ == '__main__':
    path = "/home/shenjw/3DGS/data/test/model/split_result/visible/111/partition_111.pkl"
    model_path = '/home/shenjw/3DGS/gaussian-splatting-main/partition_111/point_cloud/iteration_30000/'
    # 检查文件是否存在
    if not os.path.exists(path):
        print(f"错误：文件 {path} 不存在")
        exit(1)

    try:
        with open(path, 'rb') as f:
            partition = pickle.load(f)  # 直接加载单个分区
    except PermissionError:
        print(f"错误：没有权限读取文件 {path}")
        exit(1)
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        exit(1)

    # 调用裁剪函数处理单个分区
    cut_partition(model_path, partition)
    print("处理完成")