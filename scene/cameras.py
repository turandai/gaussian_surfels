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
import torch.nn.functional as F
import random
import numpy as np
from utils.graphics_utils import getWorld2View, getWorld2View2, getProjectionMatrix, fov2focal
from utils.general_utils import rotmat2quaternion

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, prcppoint,
                 image=torch.zeros([1, 1, 1]), 
                 gt_alpha_mask=None, image_name=None, uid=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 mask=None, mono=None, img_w=None, img_h=None, scene_scale=1, camera_lr=0):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = torch.from_numpy(R)
        self.q = rotmat2quaternion(self.R[None], True)[0]
        self.T = torch.from_numpy(T)
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.scene_scale = scene_scale

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0)#.to(self.data_device)
        self.image_width = img_w if img_w is not None else self.original_image.shape[2]
        self.image_height = img_h if img_h is not None else self.original_image.shape[1]

        # if gt_alpha_mask is not None:
        #     self.original_image *= gt_alpha_mask.to(self.data_device)
        # else:
        #     self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.q = nn.Parameter(self.q.to(torch.float32).contiguous().requires_grad_(True))
        self.T = nn.Parameter(self.T.to(torch.float32).contiguous().requires_grad_(True))
        self.to(data_device)

        self.lr = camera_lr
        l = [
            {'params': [self.q], 'lr': camera_lr * self.scene_scale},
            {'params': [self.T], 'lr': camera_lr * self.scene_scale},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.prcppoint = torch.tensor(prcppoint).to(torch.float32)#.cuda()
        self.projection_matrix = getProjectionMatrix(self.znear, self.zfar, FoVx, FoVy, self.image_width, self.image_height, prcppoint).transpose(0,1).to(data_device)
        # self.world_view_transform = None
        # self.full_proj_transform = None
        # self.camera_center = None
        self.update()

        # self.world_view_transform = getWorld2View(self.q, self.T, self.trans, self.scale).transpose(0, 1)#.cuda()
        # self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # # self.camera_center = self.world_view_transform.inverse()[3, :3].detach()
        # self.camera_center = -self.T[None]@self.world_view_transform
        # self.camera_center = -self.T.cpu()[None]@self.R.t().cpu()

        # print(self.camera_center, -self.T.cpu()[None]@self.R.t().cpu())
        # exit()

        self.mask = None if mask is None else torch.from_numpy(mask)#.to(data_device)
        self.mono = None if mono is None else torch.from_numpy(mono)#.to(data_device)
        # self.feat = None

        self.to_cpu()
        


    def update(self):
        self.world_view_transform = getWorld2View(self.q, self.T, self.trans, self.scale).transpose(0, 1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # old_center = self.camera_center
        (-self.T[None]@self.world_view_transform[:3, :3].t())[0]
        self.camera_center = (-self.T[None]@self.world_view_transform[:3, :3].t())[0]
        # if old_center is not None:
        #     print(torch.mean(old_center - self.camera_center))
        # print(self.world_view_transform.device, self.full_proj_transform.device, self.camera_center.device)
    
    def to_device(self, device=None):
        device = self.data_device if device is None else device
        attr_dict = vars(self)
        tensor_keys = [k for k, v in attr_dict.items() if type(v) == torch.Tensor]
        for k in tensor_keys:
            attr_dict[k] = attr_dict[k].to(device) # move tensors to device
        self.to(device) # move parameters to device
        self.device = self.data_device
        return self
    
    def to_cpu(self):
        attr_dict = vars(self)
        tensor_keys = [k for k, v in attr_dict.items() if type(v) == torch.Tensor]
        for k in tensor_keys:
            attr_dict[k] = attr_dict[k].cpu()
        self.cpu()
        self.device = torch.device('cpu')
        return self
    # def get_feat(self, extractor=None):
    #     if extractor is None:
    #         return self.original_image
    #     return extractor.extract_feature(self.original_image[None])[0]
            
    def get_feat(self, func=None):
        if func is None:
            return self.original_image[None]
        
        # if self.feat is None:
        #     self.feat = func(self.original_image[None])
        #     self.feat = [i.detach() for i in self.feat]
        # return self.feat

        if self.feat is None:
            feat = func(self.original_image[None])
            feat = [i.detach() for i in feat]
            self.feat = feat
            # print('generating feats')
            
        return self.feat
    
    def get_intrinsic(self):
        fx = fov2focal(self.FoVx, self.image_width)
        fy = fov2focal(self.FoVy, self.image_height)
        cx = self.prcppoint[0] * self.image_width
        cy = self.prcppoint[1] * self.image_height
        return torch.tensor([fx, 0, cx,
                             0, fy, cy,
                             0,  0, 1]).reshape([3, 3])


    def get_gtMask(self, with_mask=True):
        if self.mask is None or not with_mask:
            self.mask = torch.ones_like(self.original_image[:1], device="cuda")
        return self.mask#.to(torch.bool)

    def get_gtImage(self, bg, with_mask=True, mask_overwrite=None):
        # print(self.original_image.dtype, self.mask.dtype, self.mask.mean())
        # test = self.original_image * self.mask + bg[:, None, None] * (1 - self.mask)
        # print(test.dtype)
        if self.mask is None or not with_mask:
            return self.original_image
        # exit()
        mask = self.get_gtMask(with_mask) if mask_overwrite is None else mask_overwrite
        return self.original_image * mask + bg[:, None, None] * (1 - mask)
    
    def random_patch(self, h_size=float('inf'), w_size=float('inf')):
        h = self.image_height
        w = self.image_width
        h_size = min(h_size, h)
        w_size = min(w_size, w)
        h0 = random.randint(0, h - h_size)
        w0 = random.randint(0, w - w_size)
        h1 = h0 + h_size
        w1 = w0 + w_size
        return torch.tensor([h0, w0, h1, w1]).to(torch.float32).to(self.device)
    
    def add_noise(self, s):
        T = self.T + (random.random() - 0.5) * s
        self.T = nn.Parameter(T.contiguous().requires_grad_(True))
        self.update()

    def overwrite_pose(self, camera):
        self.q = nn.Parameter(camera.q.requires_grad_(True))
        self.T = nn.Parameter(camera.T.requires_grad_(True))
        # self.update()



                


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

