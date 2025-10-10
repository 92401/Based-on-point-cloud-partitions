import os
import pickle
from typing import NamedTuple
import numpy as np
from shapely.geometry.polygon import Polygon
import open3d as o3d
import shen_partation_densify as shen_partition_new #二叉树分区新分区代码
from utils.camera_utils import cameraList_from_camInfos_partition
from scene.ptgs.shen_data_read import partition
import sys
origin_box = []
pcd=o3d.io.read_point_cloud(self.ply)
points = np.asarray(pcd.points)[:, :2]
partition1=partition(partition.partition_id,
            origin_box=[],
            camera=[],
            extend_rate=expansion_rate,
            extend_box=expanded_bounds,
            point_num=partition.point_num,
            point_cloud=pcd_i)