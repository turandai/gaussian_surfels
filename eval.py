#  NeuralWarp  All rights reseved to Thales LAS and ENPC.
#
#  This code is freely available for academic use only and Provided “as is” without any warranty.
#
#  Modification are allowed for academic research provided that the following conditions are met :
#    * Redistributions of source code or any format must retain the above copyright notice and this list of conditions.
#    * Neither the name of Thales LAS and ENPC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# adapted from https://github.com/jzhangbs/DTUeval-python

import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import trimesh
from plyfile import PlyData, PlyElement
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.io import IO
from argparse import ArgumentParser
import sys

def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1 + 1, :n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q

def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)

def read_ply(file_path):
    ply = PlyData.read(file_path)
    ply = ply['vertex']

    vtx = np.concatenate([[ply['x']], [ply['y']], [ply['z']]], axis=0).T
    # attr = np.concatenate([[ply['nx']], [ply['ny']], [ply['nz']],
    #                        [ply['red']], [ply['green']], [ply['blue']]], axis=0).T
    attr = np.concatenate([[ply['nx']], [ply['ny']], [ply['nz']],
                           [ply['red']], [ply['green']], [ply['blue']]], axis=0).T

    # print(np.sum(attr[:, :3]))
    # exit()
    # attr[:, 3:] = (attr[:, :3] + 1) / 2 * 255
    attr[:, 3:6] /= 255

    return vtx, attr

def eval_simple(in_file, stl_file, scale):
    data_mesh = o3d.io.read_triangle_mesh(str(in_file))


    data_mesh.remove_unreferenced_vertices()

    mp.freeze_support()

    # default dtu values
    max_dist = 20 / scale
    patch = 60
    thresh = 0.2  # downsample density

    pbar = tqdm(total=4)
    pbar.set_description('read data mesh')

    vertices = np.asarray(data_mesh.vertices)

    triangles = np.asarray(data_mesh.triangles)
    tri_vert = vertices[triangles]

    pbar.update(1)
    pbar.set_description('sample pcd from mesh')
    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    non_zero_area = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [
        arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
    ]
    thr = thresh * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)

    with mp.Pool() as mp_pool:
        new_pts = mp_pool.map(sample_single_tri,
                              ((n1[i, 0], n2[i, 0], v1[i:i + 1], v2[i:i + 1], tri_vert[i:i + 1, 0]) for i in
                               range(len(n1))), chunksize=1024)

    new_pts = np.concatenate(new_pts, axis=0)
    data_pcd = np.concatenate([vertices, new_pts], axis=0)

    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)

    pbar.update(1)
    pbar.set_description('read STL pcd')
    stl_pcd = o3d.io.read_point_cloud(stl_file)
    stl = np.asarray(stl_pcd.points)

    pbar.update(1)
    pbar.set_description('compute data2stl')
    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_pcd, n_neighbors=1, return_distance=True)

    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    pbar.update(1)
    pbar.set_description('compute stl2data')

    nn_engine.fit(data_pcd)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    over_all = (mean_d2s + mean_s2d) / 2 * scale


    return over_all

def eval(in_file, scene, dataset_dir, S, T):
    data_mesh = o3d.io.read_triangle_mesh(str(in_file))


    data_mesh.remove_unreferenced_vertices()

    mp.freeze_support()

    # default dtu values
    max_dist = 20
    patch = 60
    thresh = 0.2  # downsample density

    pbar = tqdm(total=8)
    pbar.set_description('read data mesh')

    vertices = np.asarray(data_mesh.vertices)

    vertices = vertices / S + T

    triangles = np.asarray(data_mesh.triangles)
    tri_vert = vertices[triangles]

    pbar.update(1)
    pbar.set_description('sample pcd from mesh')
    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    non_zero_area = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [
        arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
    ]
    thr = thresh * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)

    with mp.Pool() as mp_pool:
        new_pts = mp_pool.map(sample_single_tri,
                              ((n1[i, 0], n2[i, 0], v1[i:i + 1], v2[i:i + 1], tri_vert[i:i + 1, 0]) for i in
                               range(len(n1))), chunksize=1024)

    new_pts = np.concatenate(new_pts, axis=0)
    data_pcd = np.concatenate([vertices, new_pts], axis=0)

    pbar.update(1)
    pbar.set_description('random shuffle pcd index')
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    pbar.update(1)
    pbar.set_description('downsample pcd')
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]

    pbar.update(1)
    pbar.set_description('masking data pcd')
    obs_mask_file = loadmat(f'{dataset_dir}/ObsMask/ObsMask{scene}_10.mat')
    ObsMask, BB, Res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
    BB = BB.astype(np.float32)

    inbound = ((data_down >= BB[:1] - patch) & (data_down < BB[1:] + patch * 2)).sum(axis=-1) == 3
    data_in = data_down[inbound]

    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) == 3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:, 0], data_grid_in[:, 1], data_grid_in[:, 2]].astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]

    pbar.update(1)
    pbar.set_description('read STL pcd')
    stl_pcd = o3d.io.read_point_cloud(f'{dataset_dir}/Points/stl/stl{scene:03}_total.ply')
    stl = np.asarray(stl_pcd.points)

    pbar.update(1)
    pbar.set_description('compute data2stl')
    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)

    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    pbar.update(1)
    pbar.set_description('compute stl2data')
    ground_plane = loadmat(f'{dataset_dir}/ObsMask/Plane{scene}.mat')['P']

    stl_hom = np.concatenate([stl, np.ones_like(stl[:, :1])], -1)
    above = (ground_plane.reshape((1, 4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]
    # stl_above = stl

    nn_engine.fit(data_in)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    pbar.update(1)
    pbar.close()
    over_all = (mean_d2s + mean_s2d) / 2

    return over_all

def eval_dtu(source_path, scanId, dtu_gt_path, in_mesh):
    scale_mat = np.load(f'{source_path}/cameras.npz')['scale_mat_0']
    S = np.linalg.inv(scale_mat[:3, :3])[0][0]
    T = scale_mat[:3, 3:].T
    print(f'Evaluating: {in_mesh}')
    cd = eval(in_mesh, scanId, dtu_gt_path, S, T)
    print('CD:', cd)
    return cd

def eval_bmvs(source_path, in_mesh):
    print(f'Evaluating: {in_mesh}')
    gt_path = f'{source_path}/gt_pts.ply'
    cd = eval_simple(in_mesh, gt_path, scale=1e3)
    print('CD:', cd)
    return cd

if __name__ == '__main__':
    parser = ArgumentParser(description="eval script parameters")
    parser.add_argument("--dataset", choices=['dtu', 'bmvs'], type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--dtu_gt_path", type=str)
    parser.add_argument("--dtu_scanId", type=int)
    args = parser.parse_args(sys.argv[1:])
    if args.dataset == 'dtu':
        eval_dtu(args.source_path, args.dtu_scanId, args.dtu_gt_path, args.mesh_path)
    elif args.dataset == 'bmvs':
        eval_bmvs(args.source_path, args.mesh_path)
