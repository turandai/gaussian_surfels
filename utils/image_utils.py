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
import numpy as np
from utils.graphics_utils import fov2focal, focal2fov
from torch.utils.cpp_extension import load
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import glob
import cv2
from tqdm import tqdm

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def depth2rgb(depth, mask):
    sort_d = torch.sort(depth[mask.to(torch.bool)])[0]
    min_d = sort_d[len(sort_d) // 100 * 5]
    max_d = sort_d[len(sort_d) // 100 * 95]
    # min_d = 2.8
    # max_d = 4.6
    # print(min_d, max_d)
    depth = (depth - min_d) / (max_d - min_d) * 0.9 + 0.1
    
    viridis = ListedColormap(plt.cm.viridis(np.linspace(0, 1, 256)))
    depth_draw = viridis(depth.detach().cpu().numpy()[0])[..., :3]
    # print(viridis(depth.detach().cpu().numpy()).shape, depth_draw.shape, mask.shape)
    depth_draw = torch.from_numpy(depth_draw).to(depth.device).permute([2, 0, 1]) * mask

    return depth_draw

def normal2rgb(normal, mask):
    normal_draw = torch.cat([normal[:1], -normal[1:2], -normal[2:]])
    normal_draw = (normal_draw * 0.5 + 0.5) * mask
    return normal_draw

def depth2normal(depth, mask, camera):
    # conver to camera position
    camD = depth.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0])
    shape = camD.shape
    device = camD.device
    h, w, _ = torch.meshgrid(torch.arange(0, shape[0]), torch.arange(0, shape[1]), torch.arange(0, shape[2]), indexing='ij')
    # print(h)
    h = h.to(torch.float32).to(device)
    w = w.to(torch.float32).to(device)
    p = torch.cat([w, h], axis=-1)
    
    p[..., 0:1] -= camera.prcppoint[0] * camera.image_width
    p[..., 1:2] -= camera.prcppoint[1] * camera.image_height
    p *= camD
    K00 = fov2focal(camera.FoVy, camera.image_height)
    K11 = fov2focal(camera.FoVx, camera.image_width)
    K = torch.tensor([K00, 0, 0, K11]).reshape([2,2])
    Kinv = torch.inverse(K).to(device)
    # print(p.shape, Kinv.shape)
    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)

    # padded = mod.contour_padding(camPos.contiguous(), mask.contiguous(), torch.zeros_like(camPos), filter_size // 2)
    # camPos = camPos + padded
    p = torch.nn.functional.pad(camPos[None], [0, 0, 1, 1, 1, 1], mode='replicate')
    mask = torch.nn.functional.pad(mask[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode='replicate').to(torch.bool)
    

    p_c = (p[:, 1:-1, 1:-1, :]      ) * mask[:, 1:-1, 1:-1, :]
    p_u = (p[:,  :-2, 1:-1, :] - p_c) * mask[:,  :-2, 1:-1, :]
    p_l = (p[:, 1:-1,  :-2, :] - p_c) * mask[:, 1:-1,  :-2, :]
    p_b = (p[:, 2:  , 1:-1, :] - p_c) * mask[:, 2:  , 1:-1, :]
    p_r = (p[:, 1:-1, 2:  , :] - p_c) * mask[:, 1:-1, 2:  , :]

    n_ul = torch.cross(p_u, p_l)
    n_ur = torch.cross(p_r, p_u)
    n_br = torch.cross(p_b, p_r)
    n_bl = torch.cross(p_l, p_b)

    # n_ul = torch.nn.functional.normalize(torch.cross(p_u, p_l), dim=-1)
    # n_ur = torch.nn.functional.normalize(torch.cross(p_r, p_u), dim=-1)
    # n_br = torch.nn.functional.normalize(torch.cross(p_b, p_r), dim=-1)
    # n_bl = torch.nn.functional.normalize(torch.cross(p_l, p_b), dim=-1)

    # n_ul = torch.nn.functional.normalize(torch.cross(p_l, p_u), dim=-1)
    # n_ur = torch.nn.functional.normalize(torch.cross(p_u, p_r), dim=-1)
    # n_br = torch.nn.functional.normalize(torch.cross(p_r, p_b), dim=-1)
    # n_bl = torch.nn.functional.normalize(torch.cross(p_b, p_l), dim=-1)
    
    n = n_ul + n_ur + n_br + n_bl
    n = n[0]
    
    # n *= -torch.sum(camVDir * camN, -1, True).sign() # no cull back

    mask = mask[0, 1:-1, 1:-1, :]

    # n = gaussian_blur(n, filter_size, 1) * mask

    n = torch.nn.functional.normalize(n, dim=-1)
    # n[..., 1] *= -1
    # n *= -1

    n = (n * mask).permute([2, 0, 1])
    return n

def normal2curv(normal, mask):
    # normal = normal.detach()
    n = normal.permute([1, 2, 0])
    m = mask.permute([1, 2, 0])
    n = torch.nn.functional.pad(n[None], [0, 0, 1, 1, 1, 1], mode='replicate')
    m = torch.nn.functional.pad(m[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode='replicate').to(torch.bool)
    n_c = (n[:, 1:-1, 1:-1, :]      ) * m[:, 1:-1, 1:-1, :]
    n_u = (n[:,  :-2, 1:-1, :] - n_c) * m[:,  :-2, 1:-1, :]
    n_l = (n[:, 1:-1,  :-2, :] - n_c) * m[:, 1:-1,  :-2, :]
    n_b = (n[:, 2:  , 1:-1, :] - n_c) * m[:, 2:  , 1:-1, :]
    n_r = (n[:, 1:-1, 2:  , :] - n_c) * m[:, 1:-1, 2:  , :]
    curv = (n_u + n_l + n_b + n_r)[0]
    curv = curv.permute([2, 0, 1]) * mask
    curv = curv.norm(1, 0, True)
    return curv

def resize_image(img, factor, mode='bilinear'):
    # print(type(img))
    if factor == 1:
        return img
    is_np = type(img) == np.ndarray
    if is_np:
        resize = torch.from_numpy(img)
    else:
        resize = img.clone()
    dtype = resize.dtype

    if type(factor) == int:
        resize = torch.nn.functional.interpolate(resize[None].to(torch.float32), scale_factor=1/factor, mode=mode)[0].to(dtype)
    elif len(factor) == 2:
        resize = torch.nn.functional.interpolate(resize[None].to(torch.float32), size=factor, mode=mode)[0].to(dtype)
    # else:

    if is_np:
        resize = resize.numpy()
    # print(type(img))
    return resize

def linear_match(d0, d1, mask, patch_size):
    # copy from MonoSDF: https://github.com/autonomousvision/monosdf/
    d0 = d0.detach()
    d1 = d1.detach()
    mask = mask.detach()

    patch_dim = (torch.tensor(d0.shape[1:3]) / patch_size).to(torch.int32)
    patch_num = patch_dim[0] * patch_dim[1]

    comb = torch.cat([d0, d1, mask], 0)
    comb_ = comb[:, :patch_dim[0] * patch_size, :patch_dim[1] * patch_size]
    comb_ = comb_.reshape([3, patch_dim[0], patch_size, patch_dim[1], patch_size])
    comb_ = comb_.permute([0, 1, 3, 2, 4])
    comb_ = comb_.reshape([3, patch_num, patch_size, patch_size])

    d0_ = comb_[0]
    d1_ = comb_[1]
    mask_ = comb_[2]
    a_00 = torch.sum(mask_ * d0_ * d0_, (1, 2))
    a_01 = torch.sum(mask_ * d0_, (1, 2))
    a_11 = torch.sum(mask_, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask_ * d0_ * d1_, (1, 2))
    b_1 = torch.sum(mask_ * d1_, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]


    d0_ = x_0[:, None, None] * d0_ + x_1[:, None, None]
    d0_ = d0_.reshape([1, patch_dim[0], patch_dim[1], patch_size, patch_size])
    d0_ = d0_.permute([0, 1, 3, 2, 4])
    d0_ = d0_.reshape([1, patch_dim[0] * patch_size, patch_dim[1] * patch_size])
    d0_b = d0[:, patch_dim[0] * patch_size:, :patch_dim[1] * patch_size]
    d0_ = torch.cat([d0_, d0_b], 1)
    d0_r = d0[:, :, patch_dim[1] * patch_size:]
    d0_ = torch.cat([d0_, d0_r], 2)
    return d0_

def match_depth(d0, d1, mask, patch_size, reso):
    clip_size = min(reso) // patch_size * patch_size
    if min(reso) % patch_size == 0:
        clip_size -= patch_size // 2

    y_ = np.random.randint(0, reso[0] - clip_size + 1)
    x_ = np.random.randint(0, reso[1] - clip_size + 1)

    d0_ = d0[:, y_:y_ + clip_size, x_:x_ + clip_size]
    d1_ = d1[:, y_:y_ + clip_size, x_:x_ + clip_size]
    mask_ = (mask)[:, y_:y_ + clip_size, x_:x_ + clip_size]

    monoD_match_ = linear_match(d0_, d1_, mask_, patch_size)

    monoD_match = d0.clone()
    monoD_match[:, y_:y_ + clip_size, x_:x_ + clip_size] = monoD_match_
    mask_match = mask.clone()
    mask_match[:, y_:y_ + clip_size, x_:x_ + clip_size] = mask_
    return monoD_match, mask_match

def depth2wpos(depth, mask, camera):
    camD = depth.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0])
    shape = camD.shape
    device = camD.device
    h, w, _ = torch.meshgrid(torch.arange(0, shape[0]), torch.arange(0, shape[1]), torch.arange(0, shape[2]), indexing='ij')
    h = h.to(torch.float32).to(device)
    w = w.to(torch.float32).to(device)
    p = torch.cat([w, h], axis=-1)
    
    p[..., 0:1] -= camera.prcppoint[0] * camera.image_width
    p[..., 1:2] -= camera.prcppoint[1] * camera.image_height
    p *= camD
    K00 = fov2focal(camera.FoVy, camera.image_height)
    K11 = fov2focal(camera.FoVx, camera.image_width)
    K = torch.tensor([K00, 0, 0, K11]).reshape([2,2])
    Kinv = torch.inverse(K).to(device)
    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)

    pose = camera.world_view_transform.to(device)
    Rinv = pose[:3, :3]
    t = pose[3:, :3]
    camWPos = (camPos - t) @ Rinv.t()

    camWPos = (camWPos[..., :3] * mask).permute([2, 0, 1])
    
    return camWPos

def depth2viewDir(depth, camera):
    camD = depth.permute([1, 2, 0])
    shape = camD.shape
    device = camD.device
    h, w, _ = torch.meshgrid(torch.arange(0, shape[0]), torch.arange(0, shape[1]), torch.arange(0, shape[2]), indexing='ij')
    h = h.to(torch.float32).to(device)
    w = w.to(torch.float32).to(device)
    p = torch.cat([w, h], axis=-1)
    
    p[..., 0:1] -= camera.prcppoint[0] * camera.image_width
    p[..., 1:2] -= camera.prcppoint[1] * camera.image_height
    p *= camD
    K00 = fov2focal(camera.FoVy, camera.image_height)
    K11 = fov2focal(camera.FoVx, camera.image_width)
    K = torch.tensor([K00, 0, 0, K11]).reshape([2,2])
    Kinv = torch.inverse(K).to(device)
    # print(p.shape, Kinv.shape)
    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)
    center = camera_center(camera)
    viewDir = camPos - center
    viewDir = torch.nn.functional.normalize(viewDir, dim=2).permute([2, 0, 1])
    return viewDir

def camera_center(camera):
    pose = camera.world_view_transform
    Rinv = pose[:3, :3]
    t = pose[3:, :3]
    center = -t @ Rinv.t()
    return center

def world2scrn(xyz, cams, pad):
    utils_mod = load(name="cuda_utils", sources=["utils/ext.cpp", "utils/cuda_utils.cu"])
    device = xyz.device
    mask = [i.get_gtMask().to(device).to(torch.float32) for i in cams]
    if pad >= 0:
        pool = torch.nn.MaxPool2d(9, stride=1, padding=pad)
    else:
        pool = torch.nn.MinPool2d(9, stride=1, padding=-pad)
    mask = pool(torch.cat(mask, 0))

    worldPos = xyz#.detach()
    worldPos = torch.cat([worldPos, torch.ones_like(worldPos[:, :1])], 1)[None, :, None]
    
    view_mat = torch.cat([i.world_view_transform[None] for i in cams], 0).to(device)[:, None]
    z_near = torch.cat([torch.tensor([[i.znear]]) for i in cams], 0).to(device)
    z_far = torch.cat([torch.tensor([[i.zfar]]) for i in cams], 0).to(device)

    camPos = (worldPos @ view_mat[..., :3]).squeeze()
    outViewZ = torch.le(camPos[..., 2], z_near) + torch.gt(camPos[..., 2], z_far)

    prj_mat = torch.cat([i.full_proj_transform[None] for i in cams], 0).to(device)[:, None]
    projPos = (worldPos @ prj_mat).squeeze()
    projPos = projPos[..., :3] / (projPos[..., 3:] + 1e-7)

    outViewX = torch.le(projPos[..., 0], -1) + torch.gt(projPos[..., 0], 1)
    outViewY = torch.le(projPos[..., 1], -1) + torch.gt(projPos[..., 1], 1)
    outView = outViewX + outViewY #+ outViewZ
    # outAllView = torch.all(outView, dim=0)

    reso = torch.cat([torch.tensor([[[i.image_width, i.image_height]]]) for i in cams], 0).to(device)
    prcp = torch.cat([i.prcppoint[None] for i in cams], 0).to(device)[:, None]

    scrnPos = ((projPos[..., :2] + 1) * reso - 1) * 0.5 + reso * (prcp - 0.5)
    ndc = (scrnPos / reso) * 2 - 1

    scrnPos = torch.clip(scrnPos, torch.zeros_like(reso), reso - 1).to(torch.long)

    mask_idx = torch.arange(0, len(mask))[:, None].to(torch.long)

    if mask.mean() == 1:
        inMask = torch.ones_like(outView).to(torch.bool)
    else:
        inMask = mask.permute([0, 2, 1])[mask_idx, scrnPos[..., 0], scrnPos[..., 1]].to(torch.bool)
    # inMaskOrOutView = torch.all(inMask + outView, dim=0)
    # inMaskOrOutView = torch.all(inMask, dim=0)

    # visible = inMaskOrOutView * ~outAllView

    return camPos, ndc, inMask, outView

def resample_points(camera, depth, normal, color, mask):
    camWPos = depth2wpos(depth, mask, camera).permute([1, 2, 0])
    camN = normal.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0]).to(torch.bool)
    mask = mask.detach()[..., 0]
    camN = camN.detach()[mask]
    camWPos = camWPos.detach()[mask]
    camRGB = color.permute([1, 2, 0])[mask]

    Rinv = camera.world_view_transform[:3, :3]

    points = torch.cat([camWPos, camN @ Rinv.t(), camRGB], -1)
    return points

def reproject_depth(depth, cam0, mask0, cami):
    wpos = depth2wpos(depth, mask0, cam0)
    wpos = wpos.reshape([3, -1]).t()[mask0.reshape([-1])]
    # print(wpos.shape)
    for i in cami:
        i.to_device()
    cpos, ndc, inMask, outView = world2scrn(wpos, cami, 0)
    ndc = ndc.permute([1, 0, 2]).permute([1, 0, 2])
    visible = (inMask * ~outView)[..., None]
    # print(ndc.shape, visible.shape)
    # exit()
    return ndc, visible

def reproject_confidence(depth, cam_cur, mask_cur, cams, extractor):
    wpos = depth2wpos(depth, mask_cur, cam_cur)
    wpos = wpos.reshape([3, -1]).t()[mask_cur.reshape([-1])]
    # print(wpos.shape)
    for i in cams:
        i.to_device()
    cpos, ndc, inMask, outView = world2scrn(wpos, cams, 0)
    ndc = ndc.permute([1, 0, 2]).permute([1, 0, 2])
    feats = [i.get_feat(extractor.extract_feats) for i in cams]
    feats = [torch.cat([i[j] for i in feats], 0) for j in range(len(feats))]

    feats_reprj = extractor.sample_feats(feats, ndc)
    confidence = extractor.fuse_feats(feats_reprj)
    # print(conf.shape, mask_cur.sum())
    print(confidence.max(), confidence.min(), confidence.mean())
    exit()
    return confidence

def cross_sample(depth, cam_cur, mask_cur, cams, feats):
    wpos = depth2wpos(depth, mask_cur, cam_cur)
    wpos = wpos.reshape([3, -1]).t()
    for i in cams:
        i.to_device()
    cpos, ndc, inMask, outView = world2scrn(wpos, cams, 0)
    ndc = ndc.permute([1, 0, 2]).permute([1, 0, 2])
    feat_cur = feats[:1]
    feat_adj = feats[1:]
    # print(feat_cur.shape, feat_adj.shape)
    # print(feats.shape, ndc[::, None].shape)
    feat_spl = torch.nn.functional.grid_sample(feat_adj, ndc[:, :, None], align_corners=False)
    visible = (inMask * ~outView)[:, None]
    feat_cur = (feat_cur * mask_cur)
    shape_spl = [feat_spl.shape[0], feat_cur.shape[1], feat_cur.shape[2], feat_cur.shape[3]]
    feat_spl = (feat_spl[..., 0] * visible * mask_cur.reshape([1, -1])).reshape(shape_spl)
    visible = visible.reshape([feat_spl.shape[0], 1, feat_cur.shape[2], feat_cur.shape[3]])
    return feat_cur, feat_spl, visible


def mask_prune(pts, cams, pad=4, batch_size=16):
    batch_num = len(cams) // batch_size + int(len(cams) % batch_size != 0)
    cams_batch = [cams[i * batch_size : min(len(cams), (i + 1) * batch_size)] for i in range(batch_num)]
    outMask = torch.zeros([len(pts)], device=pts.device, dtype=torch.bool)
    unseen = torch.ones([len(pts)], device=pts.device, dtype=torch.bool)
    for c in cams_batch:
        _, _, inMask, outView = world2scrn(pts, c, pad)
        remove = (~(inMask + outView)).any(0)
        outMask += remove
        unseen *= outView.all(0)
    remove = outMask + unseen
    return ~remove

def grid_prune(grid, shift, scale, dim, pts, thrsh=1):
    # print(dim)
    grid_cord = ((pts + shift) * scale).to(torch.long)
    # print(grid_cord.min(), grid_cord.max())
    out = (torch.le(grid_cord, 0) + torch.gt(grid_cord, dim - 1)).any(1)
    # print(grid_cord.min(), grid_cord.max())
    grid_cord = grid_cord.clamp(torch.zeros_like(dim), dim - 1)
    mask = grid[grid_cord[:, 0], grid_cord[:, 1], grid_cord[:, 2]] > thrsh
    mask *= ~out
    # print(grid_cord.shape, mask.shape, mask.sum())
    return mask.to(torch.bool)

def reflect(view, normal):
    return view - 2 * torch.sum(view * normal, 0, True) * normal

def img2video(path, fps=30):
    images = glob.glob(path+'/*.jpg') + glob.glob(path+'/*.png') + glob.glob(path+'/*.JPG')
    images.sort()
    # for i in images[:100]:
    #     print(i)
    # exit()
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'{path}/video.mp4', fourcc, fps, (width, height))
    for image in tqdm(images, desc="Creating video"):
        video.write(cv2.imread(image))
    video.release()





