# -*- coding: utf-8 -*-
#        Data: 2024-06-21 17:01
#     Project: VastGaussian
#   File Name: partition_utils.py
#      Author: KangPeilun
#       Email: 374774222@qq.com 
# Description:
import os.path

import scene
from utils.camera_utils import cameraList_from_camInfos_partition

def data_partition(lp):
    from scene.dataset_readers import sceneLoadTypeCallbacks
    from scene.ptgs.data_partition import ProgressiveDataPartitioning

    # 读取整个场景的点云以及相机，同时将相机划分为train和test
    scene_info = sceneLoadTypeCallbacks["Partition"](lp.source_path, lp.images, lp.man_trans, lp.eval, lp.llffhold)  # 得到一个场景的所有参数信息   在dataset_read中还没读
    #保存训练集和测试集相机信息
    with open(os.path.join(lp.model_path, "train_cameras.txt"), "w") as f:  #保存训练相机的信息列表
        for cam in scene_info.train_cameras:
            image_name = cam.image_name
            f.write(f"{image_name}\n")

    with open(os.path.join(lp.model_path, "test_cameras.txt"), "w") as f:  #测试相机的训练列表
        for cam in scene_info.test_cameras:
            image_name = cam.image_name
            f.write(f"{image_name}\n")

    all_cameras = cameraList_from_camInfos_partition(scene_info.train_cameras + scene_info.test_cameras, args=lp)  #转换为一个统一格式的列表  在camera_utils中还没读
    DataPartitioning = ProgressiveDataPartitioning(scene_info, all_cameras, lp.model_path,
                                                   lp.m_region, lp.n_region, lp.extend_rate, lp.visible_rate)   #创建一个分区对象
    partition_result = DataPartitioning.partition_scene   #经过可见性筛选后的场景 包括相机和点云

    # 保存每个partition的图片名称到txt文件，保存分区信息
    client = 0
    partition_id_list = []
    for partition in partition_result:
        partition_id_list.append(partition.partition_id)  #分区id
        camera_info = partition.cameras
        image_name_list = [camera_info[i].camera.image_name + '.jpg' for i in range(len(camera_info))]  #相机对应的图片
        txt_file = f"{lp.model_path}/partition_point_cloud/visible/{partition.partition_id}_camera.txt"  #当前分区的相机信息存储路径
        # 打开一个文件用于写入，如果文件不存在则会被创建
        with open(txt_file, 'w') as file:  #写了每个分区的包含的jpg
            # 遍历列表中的每个元素
            for item in image_name_list:
                # 将每个元素写入文件，每个元素占一行
                file.write(f"{item}\n")
        client += 1  #统计分区数量

    return client, partition_id_list  #分区数和分区的id


def read_camList(path):
    camList = []
    with open(path, "r") as f:
        lines = f.readlines()
        for image_name in lines:
            camList.append(image_name.replace("\n", ""))

    return camList


if __name__ == '__main__':
    read_camList(r"E:\Pycharm\3D_Reconstruct\VastGaussian\output\train_1\train_cameras.txt")