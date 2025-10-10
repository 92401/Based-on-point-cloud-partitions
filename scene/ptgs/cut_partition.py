

#处理所有分区
import os
import pickle
import numpy as np
import torch
from plyfile import PlyData
from shapely.vectorized import contains
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

def extract_point_cloud(xyz, polygon):
    """
    根据多边形边界从初始点云中筛选对应partition的点云,xz
平面提取
    """
    xy = xyz[:, [0, 1]]
    mask = contains(polygon, xy[:, 0], xy[:, 1])
    return mask

def cut_partition(model_path, partition):
    """
    裁剪分区点云并保存
    """
    save_merge_dir = os.path.join(model_path, "cutor_pcd.ply")
    oribox = partition.origin_box
    point_cloud_path = os.path.join(model_path, "point_cloud.ply")
    xyz, features_dc, features_extra, opacities, scales, rots = load_ply(point_cloud_path)
    mask = extract_point_cloud(xyz, oribox)
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

def process_all_partitions(base_path):
    """
    遍历所有分区的 pkl 文件，并裁剪点云
    """
    # 遍历 base_path 下所有子文件夹（例如 visible/000, visible/001, ...）
    sub_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(base_path, sub_dir)

        # 查找 .pkl 文件
        pkl_files = [f for f in os.listdir(sub_dir_path) if f.endswith(".pkl")]

        for pkl_file in pkl_files:
            pkl_path = os.path.join(sub_dir_path, pkl_file)

            # 提取分区名称（例如 partition_000）
            partition_name = os.path.splitext(pkl_file)[0]  # 去掉 .pkl 扩展名
            partition_dir = os.path.join(sub_dir_path, partition_name)

            # 检查分区目录是否存在
            if not os.path.exists(partition_dir):
                print(f"分区目录不存在，跳过：{partition_dir}")
                continue

            # 构造模型路径
            model_path = os.path.join(partition_dir, "output", "point_cloud", "iteration_30000")

            # 检查模型路径是否存在
            if not os.path.exists(model_path):
                print(f"模型路径不存在，跳过：{model_path}")
                continue

            # 加载 .pkl 文件
            try:
                with open(pkl_path, "rb") as f:
                    partition = pickle.load(f)
            except Exception as e:
                print(f"加载 pkl 文件失败：{pkl_path}，错误：{e}")
                continue

            # 裁剪点云
            try:
                print(f"开始裁剪：{pkl_path}")
                cut_partition(model_path, partition)
                print(f"裁剪完成：{pkl_path}\n")
            except Exception as e:
                print(f"裁剪失败：{pkl_path}，错误：{e}\n")

    print("所有分区处理完成！")
if __name__ == '__main__':
    # 主路径
    base_path = r"E:\airport_data\test\ychdata\model_all\split_result\visible"
    process_all_partitions(base_path)
    print("所有分区处理完成！")
