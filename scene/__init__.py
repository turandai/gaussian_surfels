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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
import torch
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, camera_lr: float=0.0, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}


        if os.path.exists(os.path.join(args.source_path, "sparse")):
            print("Found sparse directory, assuming COLMAP data format!")
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming NeRF data format!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "cameras.npz")):
            print("Found camera.npz file, assuming IDR data format!")
            scene_info = sceneLoadTypeCallbacks["IDR"](args.source_path, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            # with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            #     dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        print(f"Loading cameras: {len(scene_info.train_cameras)} for training and {len(scene_info.test_cameras)} for testing")
        # print('train views:', [i.uid for i in scene_info.train_cameras])
        # print('test views:', [i.uid for i in scene_info.test_cameras])
        for resolution_scale in resolution_scales:
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, self.cameras_extent, camera_lr, args)
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, self.cameras_extent, camera_lr, args)

        if gaussians is None:
            return

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        

        self.gaussians.config.append(camera_lr > 0)
        self.gaussians.config = torch.tensor(self.gaussians.config, dtype=torch.float32, device="cuda")


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        # [i.to_device() for i in self.train_cameras[scale]]
        # return [i.get() for i in self.train_cameras[scale]]
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getTrainCamerasByIdx(self, idx, scale=1.0):
        cameras = self.train_cameras[scale]
        return [cameras[i] for i in idx]
    
    def get_random_pose(self, camera0, k=4, n=1):
        # get near ref poses,
        # if i0 is None:
        #     i0 = random.randint(0, self.item_num - 1)
        centeri = torch.stack([i.world_view_transform[3, :3].cuda() for i in self.getTrainCameras()])
        center0 = camera0.world_view_transform[3, :3]
        dist = torch.abs(center0 - centeri).norm(2, -1)


        topk_v, topk_i = dist.topk(k=min(k + 1, len(centeri)), dim=0, largest=False)
        # print(topk_v)
        # print(topk_i)
        # # exit()
        topk_i = topk_i[1:].tolist()
        # print(i0, topk_i)
        random.shuffle(topk_i)
        i_n = topk_i[:min(n, k)]

        return self.getTrainCamerasByIdx(i_n)

    def get_bound(self):
        centers = torch.stack([i.camera_center for i in self.getTrainCameras()], 0)
        min_bound = centers.min(0)[0]
        max_bound = centers.max(0)[0]
        return min_bound, max_bound
    
    def visualize_cameras(self):
        points = []
        colors = []
        for i in self.getTrainCameras():
            center = i.camera_center.detach().cpu().numpy()
            # print(center)
            viewDir = i.R[:3, 2].cpu().numpy()
            for j in range(1):
                points.append(center + viewDir * j * 0.1)
                # print(center)
                # print(i.T@i.R)
                # colors.append([1, 1, 1, 1.0] if j == 0 else [0, 0, 0, 0.0])
        import pymeshlab
        import numpy as np
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.array(points)))
        ms.save_current_mesh('test/cameras.ply')
        
    def pass_pose(self, s0, s1):
        c0 = self.getTrainCameras(s0)
        c1 = self.getTrainCameras(s1)
        for i in range(len(c0)):
            c1[i].overwrite_pose(c0[i])