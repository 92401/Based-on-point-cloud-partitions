import os
import numpy as np
import torch
from plyfile import PlyData
from scene.gaussian_model import GaussianModel


def load_ply(path):
    """
    从 PLY 文件加载点的位置、SH 特征、缩放、旋转等信息。
    """
    plydata = PlyData.read(path)
    max_sh_degree = 3

    xyz = np.stack((
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"])
    ), axis=1)

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    # SH DC 分量
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    # SH 余项分量
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((xyz.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    # 缩放
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # 旋转
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return xyz, features_dc, features_extra, opacities, scales, rots


def merge_ply_files(input_folder, output_file):
    """
    合并多个高斯 .ply 文件为一个模型，并进行去重。
    """
    xyz_list = []
    features_dc_list = []
    features_extra_list = []
    opacities_list = []
    scales_list = []
    rots_list = []

    # 遍历所有 ply 文件
    for file_name in sorted(os.listdir(input_folder)):
        if file_name.endswith(".ply"):
            ply_path = os.path.join(input_folder, file_name)
            print(f"正在加载文件: {ply_path}")
            xyz, features_dc, features_extra, opacities, scales, rots = load_ply(ply_path)
            xyz_list.append(xyz)
            features_dc_list.append(features_dc)
            features_extra_list.append(features_extra)
            opacities_list.append(opacities)
            scales_list.append(scales)
            rots_list.append(rots)

    # 合并数据
    points = np.concatenate(xyz_list, axis=0)
    features_dc_list = np.concatenate(features_dc_list, axis=0)
    features_extra_list = np.concatenate(features_extra_list, axis=0)
    opacities_list = np.concatenate(opacities_list, axis=0)
    scales_list = np.concatenate(scales_list, axis=0)
    rots_list = np.concatenate(rots_list, axis=0)

    # 去重
    points, unique_indices = np.unique(points, axis=0, return_index=True)
    features_dc_list = features_dc_list[unique_indices]
    features_extra_list = features_extra_list[unique_indices]
    opacities_list = opacities_list[unique_indices]
    scales_list = scales_list[unique_indices]
    rots_list = rots_list[unique_indices]

    # 构造模型参数（保持在 CPU，节省显存）
    global_model = GaussianModel(3)
    global_params = {
        'xyz': torch.from_numpy(points).float(),
        'rotation': torch.from_numpy(rots_list).float(),
        'scaling': torch.from_numpy(scales_list).float(),
        'opacity': torch.from_numpy(opacities_list).float(),
        'features_dc': torch.from_numpy(features_dc_list).float().permute(0, 2, 1),
        'features_rest': torch.from_numpy(features_extra_list).float().permute(0, 2, 1)
    }
    global_model.set_params(global_params)
    global_model.save_ply(output_file)
    print(f"✅ 合并完成，输出文件保存至：{output_file}")


if __name__ == "__main__":
    input_folder = r"E:\airport_data\test\ychdata\model_all\split_result\visible\organized"
    output_file = r"E:\airport_data\test\ychdata\model_all\split_result\visible\merged_point_cloud.ply"
    merge_ply_files(input_folder, output_file)
