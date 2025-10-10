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


def split_gaussian_model_by_x_midpoint(input_ply, output_folder):
    """
    根据XY包围盒的X中点将高斯模型分割为左右两部分
    :param input_ply: 输入PLY文件路径
    :param output_folder: 输出文件夹
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 加载原始模型
    print(f"正在加载模型: {input_ply}")
    xyz, features_dc, features_extra, opacities, scales, rots = load_ply(input_ply)
    
    # 计算XY包围盒的X中点
    min_x = np.min(xyz[:, 0])
    max_x = np.max(xyz[:, 0])
    mid_x = (min_x + max_x) / 2
    print(f"模型X范围: {min_x:.2f} 到 {max_x:.2f}, 中点为: {mid_x:.2f}")
    
    # 根据X坐标分割点
    left_mask = xyz[:, 0] < mid_x
    right_mask = ~left_mask
    
    # 左半部分
    left_xyz = xyz[left_mask]
    left_features_dc = features_dc[left_mask]
    left_features_extra = features_extra[left_mask]
    left_opacities = opacities[left_mask]
    left_scales = scales[left_mask]
    left_rots = rots[left_mask]
    
    # 右半部分
    right_xyz = xyz[right_mask]
    right_features_dc = features_dc[right_mask]
    right_features_extra = features_extra[right_mask]
    right_opacities = opacities[right_mask]
    right_scales = scales[right_mask]
    right_rots = rots[right_mask]
    
    # 保存左半部分
    left_model = GaussianModel(3)
    left_params = {
        'xyz': torch.from_numpy(left_xyz).float(),
        'rotation': torch.from_numpy(left_rots).float(),
        'scaling': torch.from_numpy(left_scales).float(),
        'opacity': torch.from_numpy(left_opacities).float(),
        'features_dc': torch.from_numpy(left_features_dc).float().permute(0, 2, 1),
        'features_rest': torch.from_numpy(left_features_extra).float().permute(0, 2, 1)
    }
    left_model.set_params(left_params)
    left_output = os.path.join(output_folder, "left_part.ply")
    left_model.save_ply(left_output)
    print(f"✅ 左半部分已保存至: {left_output} (包含 {len(left_xyz)} 个点)")
    
    # 保存右半部分
    right_model = GaussianModel(3)
    right_params = {
        'xyz': torch.from_numpy(right_xyz).float(),
        'rotation': torch.from_numpy(right_rots).float(),
        'scaling': torch.from_numpy(right_scales).float(),
        'opacity': torch.from_numpy(right_opacities).float(),
        'features_dc': torch.from_numpy(right_features_dc).float().permute(0, 2, 1),
        'features_rest': torch.from_numpy(right_features_extra).float().permute(0, 2, 1)
    }
    right_model.set_params(right_params)
    right_output = os.path.join(output_folder, "right_part.ply")
    right_model.save_ply(right_output)
    print(f"✅ 右半部分已保存至: {right_output} (包含 {len(right_xyz)} 个点)")
    
    print(f"✅ 模型分割完成，左半部分占 {len(left_xyz)/len(xyz)*100:.1f}%，右半部分占 {len(right_xyz)/len(xyz)*100:.1f}%")


if __name__ == "__main__":
    input_ply = r"E:\airport_data\test\ychdata\model_all\split_result\visible\organized\left_part.ply"
    output_folder = r"E:\airport_data\test\ychdata\model_all\split_result\visible\split_parts"
    split_gaussian_model_by_x_midpoint(input_ply, output_folder)