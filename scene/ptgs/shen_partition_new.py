import os
import numpy as np
from plyfile import PlyData, PlyElement

# 读取PLY文件并加载点云数据
def load_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    f_dc_0 = np.asarray(plydata.elements[0]["f_dc_0"])
    f_dc_1 = np.asarray(plydata.elements[0]["f_dc_1"])
    f_dc_2 = np.asarray(plydata.elements[0]["f_dc_2"])
    features_dc = np.stack((f_dc_0, f_dc_1, f_dc_2), axis=1)

    scales = np.stack((
        np.asarray(plydata.elements[0]["scale_0"]),
        np.asarray(plydata.elements[0]["scale_1"]),
        np.asarray(plydata.elements[0]["scale_2"])
    ), axis=1)

    rots = np.stack((
        np.asarray(plydata.elements[0]["rot_0"]),
        np.asarray(plydata.elements[0]["rot_1"]),
        np.asarray(plydata.elements[0]["rot_2"]),
        np.asarray(plydata.elements[0]["rot_3"])
    ), axis=1)

    return xyz, features_dc, opacities, scales, rots

# 计算xy包围盒
def get_xy_bounding_box(xyz):
    xy = xyz[:, :2]
    min_xy = np.min(xy, axis=0)
    max_xy = np.max(xy, axis=0)
    return min_xy, max_xy

# 划分网格（四叉树：根据xy包围盒划分为四个区域）
def create_quadrants(min_xy, max_xy):
    x_mid = (min_xy[0] + max_xy[0]) / 2
    y_mid = (min_xy[1] + max_xy[1]) / 2
    
    quadrants = [
        [min_xy[0], min_xy[1], x_mid, y_mid],  # 左下
        [x_mid, min_xy[1], max_xy[0], y_mid],  # 右下
        [min_xy[0], y_mid, x_mid, max_xy[1]],  # 左上
        [x_mid, y_mid, max_xy[0], max_xy[1]]   # 右上
    ]
    return quadrants

# 根据网格筛选点云
def extract_points_in_grid(xyz, grid):
    min_x, min_y, max_x, max_y = grid
    xy = xyz[:, :2]
    mask = (xy[:, 0] >= min_x) & (xy[:, 0] <= max_x) & (xy[:, 1] >= min_y) & (xy[:, 1] <= max_y)
    return mask

# 保存分割后的点云
def save_ply(xyz, features_dc, opacities, scales, rots, save_path):
    vertex = np.core.records.fromarrays(
        [xyz[:, 0], xyz[:, 1], xyz[:, 2], features_dc[:, 0], features_dc[:, 1], features_dc[:, 2],
         opacities[:, 0], *scales.T, *rots.T],
        names='x, y, z, f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3'
    )
    vertex_element = PlyElement.describe(vertex, 'vertex', comments=["Binary data in little endian format"])
    plydata = PlyData([vertex_element], text=False)
    plydata.write(save_path)

# 递归分割模型（四叉树划分）
import gc
import numpy as np

def recursive_quadtree_cut(xyz, features_dc, opacities, scales, rots, min_xy, max_xy, max_points=2000000, level=0):
    # 计算当前网格内的点云数
    mask = extract_points_in_grid(xyz, [min_xy[0], min_xy[1], max_xy[0], max_xy[1]])
    points_in_grid = xyz[mask]
    
    # 如果点数小于最大限制，保存并返回
    if len(points_in_grid) <= max_points:
        save_dir = f"partition_level_{level}"
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f"partition_{level}.ply")
        save_ply(points_in_grid, features_dc[mask], opacities[mask], scales[mask], rots[mask], save_file)
        print(f"Saved partition {level} to {save_file}")
        return
    
    # 根据xy包围盒划分为四个子区域
    quadrants = create_quadrants(min_xy, max_xy)
    for i, g in enumerate(quadrants):
        # 清理内存
        del points_in_grid
        gc.collect()

        recursive_quadtree_cut(xyz, features_dc, opacities, scales, rots, [g[0], g[1]], [g[2], g[3]], max_points, level + 1)


# 主函数，分割模型
def cut_model_into_blocks(model_path):
    point_cloud_path = os.path.join(model_path, "point_cloud.ply")
    xyz, features_dc, opacities, scales, rots = load_ply(point_cloud_path)
    
    # 获取xy包围盒
    min_xy, max_xy = get_xy_bounding_box(xyz)
    print(f"Original bounding box: {min_xy}, {max_xy}")
    
    # 递归分割模型
    recursive_quadtree_cut(xyz, features_dc, opacities, scales, rots, min_xy, max_xy)

if __name__ == '__main__':
    model_path = r'E:\阳澄湖成果'  # 请替换为实际模型路径
    cut_model_into_blocks(model_path)
    print("Model divided into blocks.")
