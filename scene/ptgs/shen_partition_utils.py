import os
import pickle
from typing import NamedTuple
import numpy as np
from shapely.geometry.polygon import Polygon
import open3d as o3d
import shen_partation_densify as shen_partition_new #二叉树分区新分区代码
from save_partition import save_test_cameras
from utils.camera_utils import cameraList_from_camInfos_partition
from scene.ptgs.shen_data_read import partition
import sys
from utils import manhattan_utils
class ProgressiveDataPartitioning:
    # 渐进数据分区
    def __init__(self, scene_info, train_cameras, threshold,model_path): #分区的行数和列数
        self.partition_scene = None
        self.ply=scene_info.ply_path
        self.pcd = scene_info.point_cloud
        self.train_cameras = train_cameras
        self.threshold = threshold
        self.model_path = model_path
        self.partition_dir = os.path.join(model_path, "split_result")# 存放分区结果位置
        self.partition_visible_dir = os.path.join(self.partition_dir, "visible")  #可见性文件夹
        self.save_partition_data_dir = os.path.join(self.model_path, "partition_data.pkl")
        if not os.path.exists(self.partition_visible_dir): os.makedirs(self.partition_visible_dir)  # 创建 可见性相机选择后 点云的文件夹
        self.partitions= self.run_DataPartition()

    def remove_outliers(self, pcd, method="radius",
                        nb_neighbors=10, std_ratio=1,
                        radius=1.0, min_points=5):
        """
        使用 Open3D 提供的过滤函数对点云进行去噪。

        :param pcd: 输入的 open3d.geometry.PointCloud 对象
        :param method: 去噪方法，可选值 "statistical" 或 "radius"
        :param nb_neighbors: 统计滤波时计算平均距离的邻居数
        :param std_ratio: 统计滤波的标准差倍数阈值
        :param radius: 半径滤波邻域半径
        :param min_points: 半径滤波中要求的最少邻居点数
        :return: 去噪后的点云
        """
        if method == "statistical":
            # 统计离群点去除
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                     std_ratio=std_ratio)
            print(f"[StatisticalOutlierRemoval] {len(ind)} points remain after filtering.")
            return cl
        elif method == "radius":
            # 半径离群点去除
            cl, ind = pcd.remove_radius_outlier(nb_points=min_points,
                                                radius=radius)
            print(f"[RadiusOutlierRemoval] {len(ind)} points remain after filtering.")
            return cl
        else:
            raise ValueError("method 参数无效。可选值为 'statistical' 或 'radius'")

    def run_DataPartition(self):
        # Step 1: 加载点云数据
        # pcd_ori = o3d.io.read_point_cloud(self.ply)
        # #去除离群点
        # pcd= shen_partition_new.remove_statistical_outliers(pcd_ori)
        pcd_or=o3d.io.read_point_cloud(self.ply)
        #error
        pcd = self.remove_outliers(pcd_or)

        points = np.asarray(pcd.points)[:, :2]  # 只取 XY 平面数据
        # Step 2: 计算点云的边界
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)

        # 只取xz平面
        # points = np.asarray(pcd.points)[:, [0, 2]]  # 只取 XZ 平面数据
        # # 计算 XZ 平面的最小值和最大值
        # xmin, ymin = points.min(axis=0)
        # xmax, ymax = points.max(axis=0)

        bounds = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
        # Step 3: 使用四叉树进行分区
        points3d=np.asarray(pcd.points)[:, :3]
        partitions = shen_partition_new.quad_tree_partition(points3d, bounds, self.threshold)
        print(len(partitions))
        print(f"所有分区: {partitions}")
        shen_partition_new.plot_quad_tree(partitions, points, self.partition_dir+'/partitions')
        # Step 5: 可视化分区
        # 拓展
        expanded_partitions = shen_partition_new.expand_partitions(partitions, self.pcd)
        shen_partition_new.plot_quad_tree_extend(expanded_partitions, points, self.partition_dir+'/expanded_partitions')
        camera_in_expand_partitions = shen_partition_new.assign_cameras_to_partitions(expanded_partitions,self.train_cameras)
        # shen_partition.plot_subregions(camera_in_expand_partitions, self.partition_extend_dir)
        # 基于相机可视性扩展相机
        final_partitions = shen_partition_new.visibility_based_camera_selection(camera_in_expand_partitions,self.partition_visible_dir,self.pcd)
        shen_partition_new.plot_subregions(final_partitions, self.partition_visible_dir)
        return final_partitions
    def save_partition_data(self):
        """将partition后的数据序列化保存起来, 方便下次加载"""
        with open(self.save_partition_data_dir, 'wb') as f:
            pickle.dump(self.partitions, f)
    def load_partition_data(self):
        """加载partition后的数据"""
        with open(self.save_partition_data_dir, 'rb') as f:
            self.partitions = pickle.load(f)
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = r"E:\airport_data\test\ychdata"  # 默认路径
# 构建相关路径
model_path = os.path.join(path, "model")
class args(NamedTuple):
    scene_path=path
    model_path=model_path
    partition_dir=model_path
    data_device="cpu"
#曼哈顿矩阵
# pos='0 0.000000000000 0'
# rot='-90 0 0'
# man_trans = manhattan_utils.get_man_trans(pos, rot)    #获取曼哈顿的被我改了
# 读取数据
scene_partition=partition(path,None)  #运行数据读取，生成场景对象
test_cameras = cameraList_from_camInfos_partition(scene_partition.test_cameras, args)
print('测试相机',len(test_cameras))
save_test_cameras(test_cameras,path)
train_cameras = cameraList_from_camInfos_partition(scene_partition.train_cameras, args)   #将colmap的相机对象转换为sample对象，只对训练相机分区
# dict_d,distance=shen_partition_new.compute_avg_xz_distance(cameras=train_cameras, points=scene_partition.point_cloud.points)
# print('扩展距离',distance)
threshold_value =500000
DataPartitioning=ProgressiveDataPartitioning(scene_partition, train_cameras, threshold_value,model_path)




