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

from scene.cameras import Camera
import numpy as np
import torch
from utils.general_utils import PILtoTorch, quaternion2rotmat, rotmat2quaternion
from utils.graphics_utils import fov2focal
from utils.image_utils import resize_image

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, scene_scale, camera_lr):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))



    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    resized_mask = resize_image(cam_info.mask, [resolution[1], resolution[0]])
    resized_mono = None if cam_info.mono is None else resize_image(cam_info.mono, [resolution[1], resolution[0]])

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    
    # import torch
    # from torchvision.utils import save_image
    # print(cam_info.image.shape)
    # save_image(torch.from_numpy(cam_info.image).permute([2, 0, 1]) / 255, 'test/test.png')
    # exit()


    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, prcppoint=cam_info.prcppoint,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  mask=resized_mask, mono=resized_mono, scene_scale=scene_scale, camera_lr=camera_lr)

def cameraList_from_camInfos(cam_infos, resolution_scale, scene_scale, camera_lr, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, scene_scale, camera_lr))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width),
        'prcp': camera.prcppoint.tolist()
    }
    return camera_entry

def interpolate_camera(cam_lst, num):
    cam_inter = []
    count = 0
    for i in range(len(cam_lst) - 1):
        c0 = cam_lst[i]
        c0.image_name = str(count).zfill(5)
        count += 1
        cam_inter.append(c0)
        c1 = cam_lst[i + 1]
        q0 = rotmat2quaternion(c0.R[None], True)
        q1 = rotmat2quaternion(c1.R[None], True)
        # img = torch.zeros_like(c0.original_image)
        for j in range(1, num):
            k = 1 - j / num
            t = k * c0.T + (1 - k) * c1.T
            q = k * q0 + (1 - k) * q1
            R = quaternion2rotmat(torch.nn.functional.normalize(q))[0]
            fovx = k * c0.FoVx + (1 - k) * c1.FoVx
            fovy = k * c0.FoVy + (1 - k) * c1.FoVy
            prcp = k * c0.prcppoint + (1 - k) * c1.prcppoint
            c = Camera(None, R.cpu().numpy(), t.cpu().numpy(), fovx, fovy, prcp.numpy(), image_name=str(count).zfill(5),
                       img_w=c0.original_image.shape[2], img_h=c0.original_image.shape[1])
            count += 1
            cam_inter.append(c)
    cam_last = cam_lst[-1]
    cam_last.image_name = str(count).zfill(5)
    cam_inter.append(cam_last)
    return cam_inter




