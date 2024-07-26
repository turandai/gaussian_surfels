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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import torch
from torchvision.utils import save_image
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.image_utils import resize_image
from glob import glob
import imageio
import skimage
import cv2
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    prcppoint: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array
    mono: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()        
        
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width        
        
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        # print(image_path)
        image = Image.open(image_path)

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            prcppoint = np.array([intr.params[1] / width, intr.params[2] / height])
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            prcppoint = np.array([intr.params[2] / width, intr.params[3] / height])
        elif intr.model=="SIMPLE_RADIAL":
            f, cx, cy, r = intr.params
            FovY = focal2fov(f, height)
            FovX = focal2fov(f, width)
            prcppoint = np.array([cx / width, cy / height])
            # undistortion
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            D = np.array([r, 0, 0, 0])  # Only radial distortion
            image_undistorted = cv2.undistort(image_cv, K, D, None)
            image_undistorted = cv2.cvtColor(image_undistorted, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_undistorted)
        else:
            # print(intr.model, intr.params)
            # exit()
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"



        try:
            monoN = read_monoData(f'{images_folder}/../normal/{image_name}_normal.npy')
            try:
                monoD = read_monoData(f'{images_folder}/../depth/{image_name}_depth.npy')
            except FileNotFoundError:
                monoD = np.zeros_like(monoN[:1])
            mono = np.concatenate([monoN, monoD], 0)
        except FileNotFoundError:
            mono = None

        try:
            mask = load_mask(f'{images_folder}/../mask/{image_name[-3:]}.png')[None]
        except FileNotFoundError:
            mask = np.ones([1, image.size[1], image.size[0]]).astype(np.float32)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, prcppoint=prcppoint, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, mask=mask, mono=mono)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb, normal=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz) if normal is None else normal

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)

        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    try:
        with open(os.path.join(path, transformsfile)) as json_file:
            contents = json.load(json_file)
            fovx = contents["camera_angle_x"]

            frames = contents["frames"]
            for idx, frame in enumerate(frames):
                cam_name = os.path.join(path, frame["file_path"] + extension)
                # cam_name = os.path.join(path, frame["file_path"])

                transform_matrix = np.array(frame["transform_matrix"]) # c2w
                # print(transform_matrix[:, 3])
                # exit()
                # print(transform_matrix[:3, :3])

                matrix = np.linalg.inv(transform_matrix) # w2c
                R = np.transpose(matrix[:3,:3])
                T = matrix[:3, 3]
                R[:,1] *= -1
                R[:,2] *= -1
                T[1] *=- 1
                T[2] *=- 1

                # print(-T[None]@R.T)
                # exit()

                image_path = os.path.join(path, cam_name)
                image_name = Path(cam_name).stem
                image = Image.open(image_path)

                im_data = np.array(image.convert("RGBA"))

                # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

                norm_data = im_data / 255.0
                mask = norm_data[..., 3:]

                arr = norm_data[:,:,:3] #* mask + bg * (1 - mask)
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy 
                FovX = fovx
                prcppoint = np.array([0.5, 0.5])
                mask = mask.transpose([2, 0, 1]).astype(np.float32)

                prefix = transformsfile.split('.')[0].split('_')[1]
                try:
                    monoN = read_monoData(f'{path}/{prefix}_normal/{image_name}_normal.npy')
                    try:
                        monoD = read_monoData(f'{path}/{prefix}_depth/{image_name}_depth.npy')
                    except FileNotFoundError:
                        monoD = np.zeros_like(monoN[:1])
                    mono = np.concatenate([monoN, monoD], 0)
                except FileNotFoundError:
                    mono = None


                cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, prcppoint=prcppoint, image=image,
                                image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                                mask=mask, mono=mono))
    except FileNotFoundError:
        # print(os.path.join(path, transformsfile))
        print(f"{transformsfile} not found!")
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if os.path.exists(ply_path):
        pcd = fetchPly(ply_path)
        print(f"Featching points3d.ply...")
    else:
        num_pts = 50_0000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        rand_scale = 2.6
        xyz = np.random.random((num_pts, 3)) * rand_scale - rand_scale / 2
        shs = np.random.random((num_pts, 3)) / 255.0

        normal = np.random.random((num_pts, 3)) - 0.5
        normal /= np.linalg.norm(normal, 2, 1, True)
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=normal)

    #     storePly(ply_path, xyz, SH2RGB(shs) * 255, normal)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def load_rgb(path):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)

    # pixel values between [-1,1]
    img -= 0.5
    img *= 2.
    img = img.transpose(2, 0, 1)
    return img

def load_mask(path):
    alpha = imageio.imread(path, pilmode='F')
    alpha = skimage.img_as_float32(alpha) / 255
    return alpha

def read_monoData(path):
    mono = np.load(path)
    if len(mono.shape) == 4:
        mono = mono[0]
    elif len(mono.shape) == 2:
        mono = mono[None]
    return mono

def readIDRCameras(path):
    # copy from IDR: https://github.com/lioryariv/idr/

    assert os.path.exists(path), "Data directory is empty"

    image_dir = '{0}/image'.format(path)
    image_paths = sorted(glob_imgs(image_dir))
    mask_dir = '{0}/mask'.format(path)
    mask_paths = sorted(glob_imgs(mask_dir))
    cam_file = '{0}/cameras.npz'.format(path)

    n_images = len(image_paths)

    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        pose_all.append(P)


    rgb_images = []
    for i in image_paths:
        rgb = load_rgb(i)
        rgb_images.append(rgb)

    object_masks = []
    for i in mask_paths:
        object_mask = load_mask(i)
        object_masks.append(object_mask[None])
    if len(object_masks) == 0:
        object_masks = [np.ones_like(i[:1]) for i in rgb_images]

    cam_infos = []
    for i in range(n_images):
        P = pose_all[i]
        K, R, t = cv2.decomposeProjectionMatrix(P[:3, :4])[:3]
        K = K / K[2, 2]
        t = t[:3, :] / t[3:, :]
        T = -R @ t
        T = T[:, 0]
        R = R.T

        # print(R, T)
        # exit()
        
        image_path = image_paths[i]
        image_name = image_path.split('.')[0].split('/')[-1]
        uid = int(image_name.split('/')[-1])
        image = (rgb_images[i].transpose([1, 2, 0]) * 0.5 + 0.5) * 255


        try:
            monoN = read_monoData(f'{path}/normal/{image_name}_normal.npy')
            try:
                monoD = read_monoData(f'{path}/depth/{image_name}_depth.npy')
            except FileNotFoundError:
                monoD = np.zeros_like(monoN[:1])
            mono = np.concatenate([monoN, monoD], 0)
        except FileNotFoundError:
            mono = None
        
        FovY = focal2fov(K[1, 1], image.shape[0])
        FovX = focal2fov(K[0, 0], image.shape[1])
        
        if image.shape[-1] == 4:
            alpha = image[..., 3:] / 255
            object_masks[i] *= alpha.transpose([2, 0, 1])
            image = image[..., :3]
        image = Image.fromarray(np.array(image, dtype=np.byte), "RGB")
        prcppoint = K[:2, 2] / image.size[:2]
        # print(K[:2, 2], image.size[:2])
        # exit()

        cam_infos.append(CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, prcppoint=prcppoint, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                        mask = object_masks[i], mono=mono))
    return cam_infos

def readIDRSceneInfo(path, eval, testskip=8):
    cam_infos = readIDRCameras(path)

    if eval:
        # test_cams = [i for i in range(len(cam_infos)) if i % testskip == 0]
        # test split following NeuS2
        test_cams = [8, 13, 16, 21, 26, 31, 34]
        if len(cam_infos) > 56:
            test_cams.append(56)

        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx not in test_cams]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_cams]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    

    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    ply_path = os.path.join(path, "points3d.ply")
    if os.path.exists(ply_path):
        pcd = fetchPly(ply_path)
        print(f"Featching points3d.ply...")
    else:
        num_pts = 100_0000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        rand_scale = 1.2
        normal = np.random.random((num_pts, 3)) - 0.5
        normal /= np.linalg.norm(normal, 2, 1, True)
        xyz = normal * 0.5 #- rand_scale / 2

        # normal = np.repeat(np.array([[0, 1, 0.0]]), num_pts, 0)
        # normal = np.random.random((num_pts, 3)) - 0.5
        rand_scale *= 2
        xyz = np.random.random((num_pts, 3)) * rand_scale - rand_scale / 2

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=normal)

    #     storePly(ply_path, xyz, SH2RGB(shs) * 255, normal)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                        train_cameras=train_cam_infos,
                        test_cameras=test_cam_infos,
                        nerf_normalization=nerf_normalization,
                        ply_path=ply_path)
    
    return scene_info



sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "IDR": readIDRSceneInfo
}


if __name__ == '__main__':
    None
    # path = '/home/pinxuan/Desktop/point/data/dtu/scan122'
    # cam = readDTUSceneInfo(path)
    # cam = [decompose_colmap_pose(i) for i in cam]
    # print(cam[0])