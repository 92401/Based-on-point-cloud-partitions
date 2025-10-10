import os.path
import numpy as np
import pickle
import torch
from plyfile import PlyData
from scene.gaussian_model import GaussianModel
from scene.ptgs.shen_partition_new import find_adjacent_partitions


def dynamic_bounding_box(partition, partitions, skip_partition=[]):
    """
    根据分区的相邻情况动态调整边界。
    如果在某个方向没有相邻分区，则不对该方向进行边界限制。
    """
    # 获取当前分区的原始包围盒边界
    minx, miny, maxx, maxy = partition.origin_box.bounds

    # 查找相邻分区
    neighbors = find_adjacent_partitions(partition, partitions, skip_partition)

    # 判断每个方向是否有相邻分区（示例中以x方向为例，y方向类同）
    # 这里的逻辑需要根据实际地理关系来判断。举例：
    # 如果有分区的maxx在当前分区的minx附近形成公共边界，则说明有左邻域。
    # 具体如何判断左、右、上、下邻域，需要根据你的坐标系统及分区排列方式。
    # 以下只是给出一种示例性的判断逻辑：
    has_left_neighbor = False
    has_right_neighbor = False
    has_down_neighbor = False
    has_up_neighbor = False

    for n in neighbors:
        n_minx, n_miny, n_maxx, n_maxy = n.origin_box.bounds
        # 若相邻分区与本分区有公共边界在minx方向上，则有左邻居
        # 假设左邻居特征： n_maxx == minx （公共垂直线）
        if abs(n_maxx - minx) < 1e-9:
            has_left_neighbor = True
        if abs(n_minx - maxx) < 1e-9:
            has_right_neighbor = True
        if abs(n_maxy - miny) < 1e-9:
            has_down_neighbor = True
        if abs(n_miny - maxy) < 1e-9:
            has_up_neighbor = True

    # 如果没有左邻域，则左边界设为无限制（用None或np.inf表征）
    if not has_left_neighbor:
        minx = None
    # 没有右邻域
    if not has_right_neighbor:
        maxx = None
    # 没有下邻域
    if not has_down_neighbor:
        miny = None
    # 没有上邻域
    if not has_up_neighbor:
        maxy = None

    return minx, miny, maxx, maxy

def load_ply(path):  # 从point云中读取信息
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
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
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

def extract_point_cloud_mask(xyz, minx, miny, maxx, maxy):
    """
    根据给定的动态边界对点云进行过滤。
    参数中如果某个值为None，表示不对该方向进行限制。
    """
    mask = np.ones(xyz.shape[0], dtype=bool)
    if minx is not None:
        mask = mask & (xyz[:,0] >= minx)
    if maxx is not None:
        mask = mask & (xyz[:,0] <= maxx)
    if miny is not None:
        mask = mask & (xyz[:,1] >= miny)
    if maxy is not None:
        mask = mask & (xyz[:,1] <= maxy)
    return mask

def seamless_merge(model_path, partition_point_cloud_dir):
    save_merge_dir = os.path.join(partition_point_cloud_dir, "point_cloud.ply")

    # 加载partition数据
    with open(os.path.join(model_path, "partition_data.pkl"), "rb") as f:
        partition_scene = pickle.load(f)

    xyz_list = []
    features_dc_list = []
    features_extra_list = []
    opacities_list = []
    scales_list = []
    rots_list = []

    for partition in partition_scene:
        point_cloud_path = os.path.join(partition_point_cloud_dir, f"{partition.partition_id}_point_cloud.ply")
        if not os.path.exists(point_cloud_path):
            print(f"文件不存在: {point_cloud_path}，跳过该分区。")
            continue
        xyz, features_dc, features_extra, opacities, scales, rots = load_ply(point_cloud_path)

        # 动态计算分区裁剪边界
        minx, miny, maxx, maxy = dynamic_bounding_box(partition, partition_scene)

        # 根据动态边界过滤点云
        mask = extract_point_cloud_mask(xyz, minx, miny, maxx, maxy)
        xyz_list.append(xyz[mask])
        features_dc_list.append(features_dc[mask])
        features_extra_list.append(features_extra[mask])
        opacities_list.append(opacities[mask])
        scales_list.append(scales[mask])
        rots_list.append(rots[mask])

    # 拼接区域点云
    points = np.concatenate(xyz_list, axis=0)
    features_dc_list = np.concatenate(features_dc_list, axis=0)
    features_extra_list = np.concatenate(features_extra_list, axis=0)
    opacities_list = np.concatenate(opacities_list, axis=0)
    scales_list = np.concatenate(scales_list, axis=0)
    rots_list = np.concatenate(rots_list, axis=0)

    # 去重
    points, mask = np.unique(points, axis=0, return_index=True)
    features_dc_list = features_dc_list[mask]
    features_extra_list = features_extra_list[mask]
    opacities_list = opacities_list[mask]
    scales_list = scales_list[mask]
    rots_list = rots_list[mask]

    global_model = GaussianModel(3)
    global_params = {
        'xyz': torch.from_numpy(points).float().cuda(),
        'rotation': torch.from_numpy(rots_list).float().cuda(),
        'scaling': torch.from_numpy(scales_list).float().cuda(),
        'opacity': torch.from_numpy(opacities_list).float().cuda(),
        'features_dc': torch.from_numpy(features_dc_list).float().cuda().permute(0, 2, 1),
        'features_rest': torch.from_numpy(features_extra_list).float().cuda().permute(0, 2, 1)
    }
    global_model.set_params(global_params)
    global_model.save_ply(save_merge_dir)


if __name__ == '__main__':
    seamless_merge(r"E:\model",r"C:\Users\Administrator\Desktop\syk")