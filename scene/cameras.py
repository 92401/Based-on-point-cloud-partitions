#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class SimpleCamera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image_name, uid, width, height,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", points3D_ids=None):
        super(SimpleCamera, self).__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R  # 3x3 numpy array
        self.T = T  # 3-element numpy array
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_width = width
        self.image_height = height
        self.trans = trans  # Optional translation
        self.scale = scale  # Optional scaling
        self.data_device = data_device

        # 新增属性 points3D_ids
        # 如果没有提供 points3D_ids，则默认为一个空 NumPy 数组
        self.points3D_ids = points3D_ids if points3D_ids is not None else np.array([], dtype=np.int64)
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans  # Additional translation
        self.scale = scale    # Scaling factor

        # Construct the world to camera transformation matrix (4x4)
        W2C = getWorld2View2(R, T, trans, scale)  # 4x4 numpy array
        self.world_view_transform = torch.tensor(W2C, dtype=torch.float32).to(self.data_device)  # 4x4 tensor

        # Compute the camera center in world coordinates: C = -R^T T
        R_tensor = torch.tensor(R, dtype=torch.float32).to(self.data_device)  # 3x3 tensor
        T_tensor = torch.tensor(T, dtype=torch.float32).to(self.data_device)  # 3-element tensor
        self.camera_center = -torch.matmul(R_tensor.transpose(0, 1), T_tensor)  # 3-element tensor

        # Compute the projection matrix
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).to(self.data_device)  # 4x4 tensor

        # Full projection transform: World -> Camera -> Clip Space -> NDC
        self.full_proj_transform = torch.matmul(self.world_view_transform, self.projection_matrix)  # 4x4 tensor
    def get_camera_center(self):
        """
        获取相机中心在世界坐标系下的坐标。
        :return: 3-element torch.Tensor
        """
        return self.camera_center
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

