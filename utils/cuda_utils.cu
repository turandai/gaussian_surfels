#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cstdio>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include "cuda_utils.h"

#define PI 3.141592654f

struct Timer {       
      typedef CUevent_st * cudaEvent_t;
      cudaEvent_t start;
      cudaEvent_t stop;
      Timer() {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
      }
      ~Timer() {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
      }
      void Start() {
            cudaEventRecord(start, 0);
      }
      void Stop() {
            cudaEventRecord(stop, 0);
      }
      float Elapsed() {
            float elapsed;
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            return elapsed;
      }
};


__device__ float atomicCAS_f32(float *p, float cmp, float val) {
    int res = atomicCAS((int*)p, *(int*)&cmp, *(int*)&val);
    // printf("%.5f\n", *(float*)&res);
    return *(float*)&res;
}

__device__ float normalize(float* v) {
    float mod = fmaxf(sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), 0.00000001);
    v[0] /= mod;
    v[1] /= mod;
    v[2] /= mod;
    return mod;
}
__device__ double normalize(double* v) {
    float mod = fmaxf(sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), 0.00000001);
    v[0] /= mod;
    v[1] /= mod;
    v[2] /= mod;
    return mod;
}


__global__ void spatial_put_kernel(
    int* n_point,
    float* worldPos_in, 
    float* normal_in, 
    float* size_in,
    float* opac_in,
    float* group_feat,
    int* group_count,
    float* mean_feat,
    float* cube_size,
    int* cube_dim,
    float* vtx_min,
    float* R
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *n_point) return;
    int feat_dim = 6;
    float vtx_ori[3] = {worldPos_in[idx * 3 + 0], worldPos_in[idx * 3 + 1], worldPos_in[idx * 3 + 2]};
    float* normal = &normal_in[idx * 3];
    // float* size= &size_in[idx];
    // float* opac = &opac_in[idx];

    float vtx_rot[3];
    for (int i = 0; i < 3; i++) {
        vtx_rot[i] = vtx_ori[0] * R[i * 3 + 0] + vtx_ori[1] * R[i * 3 + 1] + vtx_ori[2] * R[i * 3 + 2];
    }

    int group_idx = (int)((vtx_rot[0] - vtx_min[0]) / cube_size[0]) * cube_dim[1] * cube_dim[2]
                  + (int)((vtx_rot[1] - vtx_min[1]) / cube_size[1]) * cube_dim[2]
                  + (int)((vtx_rot[2] - vtx_min[2]) / cube_size[2]);

    float* group_out = &group_feat[group_idx * feat_dim];


    atomicAdd(&group_count[group_idx], 1);
    atomicAdd(&group_out[0], vtx_rot[0]);
    atomicAdd(&group_out[1], vtx_rot[1]);
    atomicAdd(&group_out[2], vtx_rot[2]);
    atomicAdd(&group_out[3], normal[0]);
    atomicAdd(&group_out[4], normal[1]);
    atomicAdd(&group_out[5], normal[2]);



}

__global__ void spatial_get_kernel(
    int* n_point,
    float* worldPos_in, 
    float* normal_in, 
    float* size_in,
    float* opac_in,
    float* group_feat,
    int* group_count,
    float* mean_feat,
    float* cube_size,
    int* cube_dim,
    float* vtx_min,
    float* R
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *n_point) return;
    int feat_dim = 6;
    float vtx_ori[3] = {worldPos_in[idx * 3 + 0], worldPos_in[idx * 3 + 1], worldPos_in[idx * 3 + 2]};
    float* normal = &normal_in[idx * 3];
    // float* size= &size_in[idx];
    // float* opac = &opac_in[idx];

    float vtx_rot[3];
    for (int i = 0; i < 3; i++) {
        vtx_rot[i] = vtx_ori[0] * R[i * 3 + 0] + vtx_ori[1] * R[i * 3 + 1] + vtx_ori[2] * R[i * 3 + 2];
    }


    int group_idx = (int)((vtx_rot[0] - vtx_min[0]) / cube_size[0]) * cube_dim[1] * cube_dim[2]
                  + (int)((vtx_rot[1] - vtx_min[1]) / cube_size[1]) * cube_dim[2]
                  + (int)((vtx_rot[2] - vtx_min[2]) / cube_size[2]);
    if (group_count[group_idx] == 0) return;
    float* group_out = &group_feat[group_idx * feat_dim];

    float acc_vtx_rot[3] = {group_out[0], group_out[1], group_out[2]}, acc_vtx_ori[3];
    for (int i = 0; i < 3; i++) {
        acc_vtx_ori[i] = acc_vtx_rot[0] * R[i + 0] + acc_vtx_rot[1] * R[i + 3] + acc_vtx_rot[2] * R[i + 6];
    }

    float buf[6];
    for (int i = 0; i < feat_dim; i++) {
        if (i < 3) buf[i] = acc_vtx_ori[i] / group_count[group_idx];
        else buf[i] = group_out[i] / group_count[group_idx];
    }

    float mean_normal_mod = normalize(&buf[3]);

    float* mean_out = &mean_feat[idx * feat_dim];
    for (int i = 0; i < feat_dim; i++) mean_out[i] = buf[i];
}

void spatial_grouping_cuda(
    int* n_point,
    float* worldPos, 
    float* normal, 
    float* size,
    float* opac,
    float* group_feat,
    int* group_count,
    float* mean_feat,
    float* cube_size,
    int* cube_dim,
    float* vtx_min,
    float* R
) {

    const int n_thread_grouping = 512;
    int n_block_grouping = *n_point / n_thread_grouping + 1;
    // int n_block_projection =  *n_point / n_thread_projection + 1;
    
    int *n_point_cuda;
    cudaMalloc((void**) &n_point_cuda, 4);
    cudaMemcpy(n_point_cuda, n_point, 4, cudaMemcpyHostToDevice);
    // cudaMalloc((void**) &n_div_cuda, 4);
    // cudaMemcpy(n_div_cuda, n_div, 4, cudaMemcpyHostToDevice);
    // printf("%d", *n_leaf);

    spatial_put_kernel<<<n_block_grouping, n_thread_grouping>>>(
        n_point_cuda,
        worldPos,
        normal,
        size,
        opac,
        group_feat,
        group_count,
        mean_feat,
        cube_size,
        cube_dim,
        vtx_min,
        R
    );

    spatial_get_kernel<<<n_block_grouping, n_thread_grouping>>>(
        n_point_cuda,
        worldPos,
        normal,
        size,
        opac,
        group_feat,
        group_count,
        mean_feat,
        cube_size,
        cube_dim,
        vtx_min,
        R
    );


}




__global__ void contour_padding_kernel(
    float* image,
    bool* mask,
    int* reso,
    float* res,
    int* size
) {
    int thrIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // int patch_size = 16; 
    // int pchNum[2] =  {reso[0] / patch_size + 1, reso[1] / patch_size + 1}; // y, x
    // if (reso[0] % patch_size == 0) pchNum[0]--;
    // if (reso[1] % patch_size == 0) pchNum[1]--;
    // int thrPerPch = patch_size * patch_size;
    // int pchIdx = thrIdx / thrPerPch, thrInPchIdx = thrIdx % thrPerPch;
    // int pchPos[2] = {pchIdx / pchNum[1], pchIdx % pchNum[1]};
    // int thrInPchPos[2] = {thrInPchIdx / patch_size, thrInPchIdx % patch_size};
    // int pixPos[2] = {pchPos[0] * patch_size + thrInPchPos[0], pchPos[1] * patch_size + thrInPchPos[1]};

    int pixPos[2] = {thrIdx / reso[1], thrIdx % reso[1]};

    if (pixPos[0] < 0 || pixPos[0] >= reso[0] || pixPos[1] < 0 || pixPos[1] >= reso[1]) return;

    // printf("%d, %d, %d\n", thrIdx, reso[0], reso[1]);

    int pixIdx = pixPos[0] * reso[1] + pixPos[1];
    if (mask[pixIdx] == 1) return;

    // int size = 1;
    const int max_chanel = 15;
    int chanel = reso[2];
    float buffer[max_chanel + 1] = {0};//
    for (int y = pixPos[0] - *size; y < pixPos[0] + *size + 1; y++) {
        for (int x = pixPos[1] - *size; x < pixPos[1] + *size + 1; x++) {
            if (y < 0 || y >= reso[0] || x < 0 || x >= reso[1]) continue;
            int pixIdx_temp = y * reso[1] + x;
            if (mask[pixIdx_temp] == 0) continue;
            for (int c = 0; c < min(max_chanel, chanel); c++) {
                buffer[c] += image[chanel * c + pixIdx_temp];
            }
            buffer[max_chanel] += 1;

        }
    }

    if (buffer[max_chanel] > 0) {
        for (int c = 0; c < min(max_chanel, chanel); c++) {
            res[pixIdx] = buffer[c] / buffer[max_chanel];
        }
    }
    // if (mask[pixIdx] == 1) res[pixIdx] = 1;


}


void contour_padding_cuda(
    float* image,
    bool* mask,
    int* reso,
    float* res,
    int* size
) {
    int *reso_cuda, *size_cuda, *chanel_cuda;
    cudaMalloc((void**) &reso_cuda, 12);
    cudaMemcpy(reso_cuda, reso, 12, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &size_cuda, 4);
    cudaMemcpy(size_cuda, size, 4, cudaMemcpyHostToDevice);
    const int patch_size = 16, n_thread_padding = 256;
    int n_block_padding = reso[0] * reso[1] / n_thread_padding + 1;
    // printf("%d, %d\n", n_block_padding, n_thread_padding);
    contour_padding_kernel<<<n_block_padding, n_thread_padding>>>(
        image, mask, reso_cuda, res, size_cuda
    );

}

__global__ void point2tsdf_kernel(
    int* n_point,
    float* wpos_in,
    float* normal_in,
    float* size_in,
    float* config,
    float* volgrid_d,
    float* volgrid_w,
    float* grid_range_in,
    int* grid_dim_x,
    float* truncate_in,
    float* step_len_in,
    float* tangent_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *n_point) return;

    float wpos[3] = {wpos_in[idx * 3 + 0], wpos_in[idx * 3 + 1], wpos_in[idx * 3 + 2]};
    float normal[3] = {normal_in[idx * 3 + 0], normal_in[idx * 3 + 1], normal_in[idx * 3 + 2]};
    float size = size_in[idx * 3 + 0] * config[0] * sqrtf(config[2]);
    // printf("%.5f, %.5f, %.5f, %.5f\n", size_in[idx * 3 + 0], config[0], sqrtf(config[2]), size);

    // printf("%.5f, %.5f, %.5f\n", normal[0], normal[1], normal[2]);
    normalize(normal);
    // float normal_mod = sqrtf(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    // normal[0] /= normal_mod;
    // normal[1] /= normal_mod;
    // normal[2] /= normal_mod;
    // float center[3] = {2, 2, 2};
    // float normal[3] = {1 / sqrtf(3), 1 / sqrtf(3), 1 / sqrtf(3)};

    // map rand to point tangent space
    float cross[3] = {-normal[1], normal[0], 0};
    float dot = normal[2];
    float k = 1 / (1 + dot);
    float rotation[6] = {(cross[0] * cross[0] * k) + dot,       (cross[1] * cross[0] * k) - cross[2],
                         (cross[0] * cross[1] * k) + cross[2],  (cross[1] * cross[1] * k) + dot,
                         (cross[0] * cross[2] * k) - cross[1],  (cross[1] * cross[2] * k) + cross[0]};

    float truncate = *truncate_in, step_len = *step_len_in;
    float grid_range[6] = {grid_range_in[0], grid_range_in[1], grid_range_in[2],
                           grid_range_in[3], grid_range_in[4], grid_range_in[5]};
    float grid_size = (grid_range[3] - grid_range[0]) / (*grid_dim_x - 1);
    int grid_dim[3] = {
        (grid_range[3] - grid_range[0]) / grid_size + 1,
        (grid_range[4] - grid_range[1]) / grid_size + 1,
        (grid_range[5] - grid_range[2]) / grid_size + 1
    };
    

    // // printf("")
    // for (int i = 0; i < *n_rand; i++) {
    //     float rand[2] = {rand_in[i * 2 + 0], rand_in[i * 2 + 1]};
    //     if (rand[0] * rand[0] + rand[1] * rand[1] > 1) continue;
    //     // float rand[2] = {0.1, 0.2};
    //     float dir_i[3] = {rotation[0] * rand[0] + rotation[1] * rand[1],
    //                       rotation[2] * rand[0] + rotation[3] * rand[1],
    //                       rotation[4] * rand[0] + rotation[5] * rand[1]};
    //     float rand_i[3] = {wpos[0] + dir_i[0] * size, wpos[1] + dir_i[1] * size, wpos[2] + dir_i[2] * size};


    //     // for (int j = -truncate / step_len; j < truncate / step_len; j++) {
    //     for (int j = 0; j < 1; j++) {
    //         // write to volgrid
    //         float d_j = j * step_len, w_j = 1;
    //         float rand_j[3] = {rand_i[0] + normal[0] * d_j, rand_i[1] + normal[1] * d_j, rand_i[2] + normal[2] * d_j};
    //         int vol_cord[3] = {(rand_j[0] - vol_range[0]) / vol_size, (rand_j[1] - vol_range[1]) / vol_size, (rand_j[2] - vol_range[2]) / vol_size};
    //         int vol_idx = vol_cord[0] * vol_dim[1] * vol_dim[2] + vol_cord[1] * vol_dim[2] + vol_cord[2];
    //         // printf("%.5f, %.5f, %.5f, %.5f\n", (rand_j[0] - vol_range[0]), (rand_j[1] - vol_range[1]), (rand_j[2] - vol_range[2]), vol_size);
    //         // printf("%d, %d, %d, %d\n", vol_idx, vol_cord[0], vol_cord[1], vol_cord[2]);
    //         atomicAdd(&volgrid_d[vol_idx], d_j * w_j);
    //         atomicAdd(&volgrid_w[vol_idx], w_j);
            
        
    //     }

    //     float dir = dir_i[0] * normal[0] + dir_i[1] * normal[1] + dir_i[2] * normal[2];
    //     if (fabsf(dir) > 0.000001) printf("%.5f, %.5f, %.5f, %.5f\n", dir);
    // }

    // calculate bbox for tangent cylinder
    float p0[3] = {wpos[0] + normal[0] *  truncate, wpos[1] + normal[1] *  truncate, wpos[2] + normal[2] *  truncate};
    float p1[3] = {wpos[0] + normal[0] * -truncate, wpos[1] + normal[1] * -truncate, wpos[2] + normal[2] * -truncate};
    float p_min[3] = {fminf(p0[0], p1[0]), fminf(p0[1], p1[1]), fminf(p0[2], p1[2])};
    float p_max[3] = {fmaxf(p0[0], p1[0]), fmaxf(p0[1], p1[1]), fmaxf(p0[2], p1[2])};
    for (int i = 0; i < 3; i++) {
        p_min[i] = fmaxf(p_min[i] - size, grid_range[i + 0]);
        p_max[i] = fminf(p_max[i] + size, grid_range[i + 3]);
    }

    // printf("%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n", 
    // wpos[0], wpos[1], wpos[2], normal[0], normal[1], normal[2],
    // p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]);
    // printf("%.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n", p_max[0] - p_min[0], p_max[1] - p_min[1], p_max[2] - p_min[2]);
    

    for (float x = p_min[0]; x < p_max[0]; x += step_len) {
        for (float y = p_min[1]; y < p_max[1]; y += step_len) {
            for (float z = p_min[2]; z < p_max[2]; z += step_len) {
    // for (float x = p_max[0]; x > p_min[0]; x -= step_len) {
    //     for (float y = p_max[1]; y > p_min[1]; y -= step_len) {
    //         for (float z = p_max[2]; z > p_min[2]; z -= step_len) {
                // printf("%.5f, %.5f, %.5f\n", x, y, z);
                float dif2point[3] = {x - wpos[0], y - wpos[1], z - wpos[2]};
                float d2plane = normal[0] * dif2point[0] + normal[1] * dif2point[1] + normal[2] * dif2point[2];
                float d2point2 = dif2point[0] * dif2point[0] + dif2point[1] * dif2point[1] + dif2point[2] * dif2point[2];
                float d2axis = sqrtf(fmaxf(d2point2 - d2plane * d2plane, 0));
                // if (d2axis > size) continue;
                if (d2plane > grid_size) continue;
                // if (d2plane < -grid_size) continue;
                int vol_cord[3] = {(x - grid_range[0]) / grid_size, (y - grid_range[1]) / grid_size, (z - grid_range[2]) / grid_size};
                int vol_idx = vol_cord[0] * grid_dim[1] * grid_dim[2] + vol_cord[1] * grid_dim[2] + vol_cord[2];
                // printf("%d, %d, %d, %d\n", vol_idx, vol_cord[0], vol_cord[1], vol_cord[2]);
                float w = expf(d2axis / size * d2axis / size * config[2] / -2);
                // w *= 1 - fminf(1, d2plane / truncate);
                // if (w < 0.000001) continue;
                // if (isnan(d2plane) || isnan(w)) printf("%.5f, %.5f, %.5f\n", d2point2, d2plane, d2axis);
                // if (d2plane < -grid_size) continue;
                // d2plane = fabs(d2plane);
                // if (d2plane < 2 * grid_size) d2plane = 2 * d2plane - 2 * grid_size;
                // if (d2plane < 2 * grid_size) d2plane = 3 / 2 * d2plane - 1 * grid_size;
                // if (d2plane < grid_size) d2plane = 2 * d2plane - 1 * grid_size;
                // if (d2plane < 2 * grid_size) d2plane = 2 * d2plane - 2 * grid_size;
                // else d2plane = fabsf(d2plane);
                atomicAdd(&volgrid_d[vol_idx], d2plane * w);
                atomicAdd(&volgrid_w[vol_idx], w);
            }
        }
    }



}



void point2tsdf_cuda(
    int* n_point,
    float* wpos,
    float* normal,
    float* size,
    float* config,
    float* volgrid_d,
    float* volgrid_w,
    float* grid_range,
    int* grid_dim_x,
    float* truncate,
    float* step_len,
    float* tangent_size
) {

    // float size_scale = config[0] * sqrtf(config[2]);
    const int n_thread_tsdf = 512;
    int n_block_tsdf = *n_point / n_thread_tsdf + 1;
    // int n_block_projection =  *n_point / n_thread_projection + 1;
    
    int *n_point_cuda;
    cudaMalloc((void**) &n_point_cuda, 4);
    cudaMemcpy(n_point_cuda, n_point, 4, cudaMemcpyHostToDevice);
    // cudaMalloc((void**) &n_div_cuda, 4);
    // cudaMemcpy(n_div_cuda, n_div, 4, cudaMemcpyHostToDevice);
    // printf("%d", *n_leaf);

    point2tsdf_kernel<<<n_block_tsdf, n_thread_tsdf>>>(
        n_point_cuda,
        wpos, normal, size,
        config,
        volgrid_d, volgrid_w,
        grid_range, grid_dim_x,
        truncate, step_len, tangent_size
    );

    

}


__global__ void reprojection_flow_determine_kernel(
    float* wpos0,
    float* normal0,
    float* camera0,
    float* camera1,
    float* min_depth
) {
    int reso[2] = {camera1[16], camera1[17]};
    int thrIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixPos[2] = {thrIdx / reso[1], thrIdx % reso[1]};
    if (pixPos[0] < 0 || pixPos[0] >= reso[0] || pixPos[1] < 0 || pixPos[1] >= reso[1]) return;
    
    // reproject
    int pix_idx = pixPos[0] * reso[1] + pixPos[1];
    float camPos[3], worldNormal[3], camNormal[3];
    for (int i = 0; i < 3; i++) {
        worldNormal[i] = normal0[pix_idx * 3 + 0] * camera0[26 + i * 3 + 0]
                       + normal0[pix_idx * 3 + 1] * camera0[26 + i * 3 + 1]
                       + normal0[pix_idx * 3 + 2] * camera0[26 + i * 3 + 2];
    }
    for (int i = 0; i < 3; i++) {
        camPos[i] = (wpos0[pix_idx * 3 + 0] - camera1[9 ]) * camera1[i * 3 + 0]
                  + (wpos0[pix_idx * 3 + 1] - camera1[10]) * camera1[i * 3 + 1]
                  + (wpos0[pix_idx * 3 + 2] - camera1[11]) * camera1[i * 3 + 2];
        camNormal[i] = worldNormal[0] * camera1[i * 3 + 0]
                     + worldNormal[1] * camera1[i * 3 + 1]
                     + worldNormal[2] * camera1[i * 3 + 2];
    }
    int scrnPos[2];
    scrnPos[0] = (camPos[0] * camera1[21] + camPos[1] * camera1[23]) / camPos[2] + camera1[24];
    scrnPos[1] = (camPos[1] * camera1[22])                           / camPos[2] + camera1[25];
    bool cond_z = (camPos[2] > camera1[14]) && (camPos[2] < camera1[15]);
    // normalize(camNormal);
    bool back_prj = (camPos[0] * camNormal[0] + camPos[1] * camNormal[1] + camPos[2] * camNormal[2]) > 0;
    if (!cond_z) return;
    // if (back_prj) return;
    bool cond_x = (scrnPos[0] >= 0) && (scrnPos[0] < camera1[17]);
    bool cond_y = (scrnPos[1] >= 0) && (scrnPos[1] < camera1[16]);
    if (!(cond_x && cond_y)) return;


    for (int i = 0; i <= 1; i++) {
        for (int j = 0; j <= 1; j++) {
            int scrnPos_temp[2] = {scrnPos[0] + i, scrnPos[1] + j};
            int flow_buf[2], scrn_idx = (reso[0] - scrnPos_temp[1] - 1) * reso[1] + scrnPos_temp[0];
            float global_depth = min_depth[scrn_idx], thread_depth = camPos[2];

            while (global_depth == 0 or global_depth > thread_depth) {
            //     printf("%.5f, %.5f\n", global_depth, thread_depth);
                global_depth = atomicCAS_f32(&min_depth[scrn_idx], global_depth, thread_depth);
            }
            // global_depth = atomicCAS_f32(&min_depth[scrn_idx], 0, thread_depth);
        }
    }


}

__global__ void reprojection_flow_write_kernel(
    float* wpos0,
    float* normal0,
    float* camera0,
    float* camera1,
    float* min_depth,
    float* flow,
    bool* mask
) {
    int reso[2] = {camera1[16], camera1[17]};
    int thrIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixPos[2] = {thrIdx / reso[1], thrIdx % reso[1]};
    if (pixPos[0] < 0 || pixPos[0] >= reso[0] || pixPos[1] < 0 || pixPos[1] >= reso[1]) return;
    
    // reproject
    int pix_idx = pixPos[0] * reso[1] + pixPos[1];
    float camPos[3], worldNormal[3], camNormal[3];
    for (int i = 0; i < 3; i++) {
        worldNormal[i] = normal0[pix_idx * 3 + 0] * camera0[26 + i * 3 + 0]
                       + normal0[pix_idx * 3 + 1] * camera0[26 + i * 3 + 1]
                       + normal0[pix_idx * 3 + 2] * camera0[26 + i * 3 + 2];
    }
    for (int i = 0; i < 3; i++) {
        camPos[i] = (wpos0[pix_idx * 3 + 0] - camera1[9 ]) * camera1[i * 3 + 0]
                  + (wpos0[pix_idx * 3 + 1] - camera1[10]) * camera1[i * 3 + 1]
                  + (wpos0[pix_idx * 3 + 2] - camera1[11]) * camera1[i * 3 + 2];
        camNormal[i] = worldNormal[0] * camera1[i * 3 + 0]
                     + worldNormal[1] * camera1[i * 3 + 1]
                     + worldNormal[2] * camera1[i * 3 + 2];
    }
    int scrnPos[2];
    scrnPos[0] = (camPos[0] * camera1[21] + camPos[1] * camera1[23]) / camPos[2] + camera1[24];
    scrnPos[1] = (camPos[1] * camera1[22])                           / camPos[2] + camera1[25];
    bool cond_z = (camPos[2] > camera1[14]) && (camPos[2] < camera1[15]);
    // normalize(camNormal);
    bool back_prj = (camPos[0] * camNormal[0] + camPos[1] * camNormal[1] + camPos[2] * camNormal[2]) > 0;
    if (!cond_z) return;
    // if (back_prj) return;
    bool cond_x = (scrnPos[0] >= 0) && (scrnPos[0] < camera1[17]);
    bool cond_y = (scrnPos[1] >= 0) && (scrnPos[1] < camera1[16]);
    if (!(cond_x && cond_y)) return;

    for (int i = 0; i <= 1; i++) {
        for (int j = 0; j <= 1; j++) {
            int scrnPos_temp[2] = {scrnPos[0] + i, scrnPos[1] + j};
            int flow_buf[2], scrn_idx = (reso[0] - scrnPos_temp[1] - 1) * reso[1] + scrnPos_temp[0];
            float global_depth = min_depth[scrn_idx], thread_depth = camPos[2], fuzzy = 0;
            if (global_depth >= thread_depth * (1 - fuzzy) && global_depth <= thread_depth * (1 + fuzzy)) {
                flow[scrn_idx * 2 + 0] = (float)pixPos[1] / (float)reso[1] * 2 - 1;
                flow[scrn_idx * 2 + 1] = (float)pixPos[0] / (float)reso[0] * 2 - 1;
                // printf("%.5f, %.5f\n", flow[scrn_idx * 2 + 0], flow[scrn_idx * 2 + 1]);
                mask[scrn_idx] = 1;
            }
        }
    }

}



void reprojection_flow_cuda(
    float* worldPos0,
    float* normal0,
    float* camera0,
    float* camera1,
    float* min_depth,
    float* flow,
    bool* mask
) {
    float reso[2];
    cudaMemcpy(reso, &camera1[16], 8, cudaMemcpyDeviceToHost);
    const int n_thread_shading = 256;
    int n_block_shading = (int)reso[0] * (int)reso[1] / n_thread_shading + 1;

    reprojection_flow_determine_kernel<<<n_block_shading, n_thread_shading>>>(
        worldPos0, normal0, camera0, camera1, min_depth
    );
    reprojection_flow_write_kernel<<<n_block_shading, n_thread_shading>>>(
        worldPos0, normal0, camera0, camera1, min_depth, flow, mask
    );
}







void visual_hull_cuda(
    float* grid,
    bool* mask,
    float* camera,
    float* config
) {
    


}


// __global__ void integrate(
//     float * tsdf_vol,
//     float * weight_vol,
//     float * color_vol,
//     float * vol_dim,
//     float * vol_origin,
//     float * cam_intr,
//     float * cam_pose,
//     float * other_params,
//     float * color_im,
//     float * depth_im) {
//     // Get voxel index
//     int gpu_loop_idx = (int) other_params[0];
//     int max_threads_per_block = blockDim.x;
//     int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
//     int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
//     int vol_dim_x = (int) vol_dim[0];
//     int vol_dim_y = (int) vol_dim[1];
//     int vol_dim_z = (int) vol_dim[2];
//     if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
//         return;
//     // Get voxel grid coordinates (note: be careful when casting)
//     float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
//     float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
//     float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);
//     // Voxel grid coordinates to world coordinates
//     float voxel_size = other_params[1];
//     float pt_x = vol_origin[0]+voxel_x*voxel_size;
//     float pt_y = vol_origin[1]+voxel_y*voxel_size;
//     float pt_z = vol_origin[2]+voxel_z*voxel_size;
//     // World coordinates to camera coordinates
//     float tmp_pt_x = pt_x-cam_pose[0*4+3];
//     float tmp_pt_y = pt_y-cam_pose[1*4+3];
//     float tmp_pt_z = pt_z-cam_pose[2*4+3];
//     float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
//     float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
//     float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;
//     // Camera coordinates to image pixels
//     int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
//     int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);
//     // Skip if outside view frustum
//     int im_h = (int) other_params[2];
//     int im_w = (int) other_params[3];
//     if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z<0)
//         return;
//     // Skip invalid depth
//     float depth_value = depth_im[pixel_y*im_w+pixel_x];
//     if (depth_value == 0)
//         return;
//     // Integrate TSDF
//     float trunc_margin = other_params[4];
//     float depth_diff = depth_value-cam_pt_z;
//     if (depth_diff < -trunc_margin)
//         return;
//     float dist = fmin(1.0f,depth_diff/trunc_margin);
//     float w_old = weight_vol[voxel_idx];
//     float obs_weight = other_params[5];
//     float w_new = w_old + obs_weight;
//     weight_vol[voxel_idx] = w_new;
//     tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx]*w_old+obs_weight*dist)/w_new;
    
//     // Integrate color
//     return;
//     float old_color = color_vol[voxel_idx];
//     float old_b = floorf(old_color/(256*256));
//     float old_g = floorf((old_color-old_b*256*256)/256);
//     float old_r = old_color-old_b*256*256-old_g*256;
//     float new_color = color_im[pixel_y*im_w+pixel_x];
//     float new_b = floorf(new_color/(256*256));
//     float new_g = floorf((new_color-new_b*256*256)/256);
//     float new_r = new_color-new_b*256*256-new_g*256;
//     new_b = fmin(roundf((old_b*w_old+obs_weight*new_b)/w_new),255.0f);
//     new_g = fmin(roundf((old_g*w_old+obs_weight*new_g)/w_new),255.0f);
//     new_r = fmin(roundf((old_r*w_old+obs_weight*new_r)/w_new),255.0f);
//     color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;
// }

__device__ float* quaternion2rotmat(float* q) {
	// Normalize quaternion to get valid rotation
	// glm::vec4 q = rot;// / glm::length(rot);
	float r = q[0], x = q[1], y = q[2], z = q[3];

	// Compute rotation matrix from quaternion
	float R[9] = {
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    };
	return R;
}

// __global__ void gaussian2occgrid_kernel(
    // int* n_point
    // float* pos_min,
    // float* pos_max,
    // float* grid_len,
    // int* grid_dim,
    // float* grid,
    // float* pos_in,
    // float* rot_in,
    // float* opac_in
// ) {
    // if (blockIdx.x == 0) printf("haha");
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx >= *n_point) return;
    // opac_in[idx] = 1;
    // float *pos = &pos_in[idx * 3], *rot = &rot_in[idx * 4], opac = opac_in[idx];
    // float *R = quaternion2rotmat(rot);
    // printf("\n%.5f, %.5f, %.5f, \n%.5f, %.5f, %.5f, \n%.5f, %.5f, %.5f\n",
    // R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8], R[9]);


// }

// void gaussian2occgrid_cuda(
//     int* n_point,
//     float* pos_min,
//     float* pos_max,
//     float* grid_len,
//     int* grid_dim,
//     float* grid,
//     float* pos,
//     float* rot,
//     float* opac
// ) {
//     const int n_thread = 512;
//     int n_block = *n_point / n_thread + 1;


//     gaussian2occgrid_kernel<<<n_block, n_thread>>>(
//         // n_point_cuda
//         // pos_min, pos_max,
//         // grid_len, grid_dim, grid,
//         // pos, rot, opac
//     );
//     // printf("haha");



// }

__global__ void gaussian2occgrid_kernel(
    int* n_point,
    float* pos_min,
    float* pos_max,
    float* grid_len,
    int* grid_dim,
    float* grid,
    float* pos_in,
    float* rot_in,
    float* scale_in,
    float* opac_in,
    float* cutoff_in
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *n_point) return;
    float pos[3] = {pos_in[idx * 3 + 0], pos_in[idx * 3 + 1], pos_in[idx * 3 + 2]},
          scale[3] = {scale_in[idx * 3 + 0], scale_in[idx * 3 + 1], scale_in[idx * 3 + 2]},
          rot[4] = {rot_in[idx * 4 + 0], rot_in[idx * 4 + 1], rot_in[idx * 4 + 2], rot_in[idx * 4 + 3]},
          opac = opac_in[idx], cutoff = fmaxf(*cutoff_in, 1.0f / 255.0f),
          pmin[3] = {pos_min[0], pos_min[1], pos_min[2]}, pmax[3] = {pos_max[0], pos_max[1], pos_max[2]};
    int gdim[3] = {grid_dim[0], grid_dim[1], grid_dim[2]};
    float *R = quaternion2rotmat(rot);
    float dir_x[3] = {R[0], R[3], R[6]}, dir_y[3] = {R[1], R[4], R[7]}, normal[3] = {R[2], R[5], R[8]};
    float grid_step = *grid_len, search_cutoff_v = 1.0f / 255.0f / opac, max_scale = fmaxf(scale[0], scale[1]);
    float search_cutoff_d2 = -2 * logf(search_cutoff_v);
    float search_radii = max_scale * sqrtf(search_cutoff_d2) + grid_step * 0.5, search_step = 0.5 * grid_step;

    for (float x = pos[0] - search_radii; x < pos[0] + search_radii; x += search_step) {
        for (float y = pos[1] - search_radii; y < pos[1] + search_radii; y += search_step) {
            for (float z = pos[2] - search_radii; z < pos[2] + search_radii; z += search_step) {
                if (x < pmin[0] || y < pmin[1] || z < pmin[2] || x >= pmax[0] || y >= pmax[1] || z >= pmax[2]) return;
                float dif[3] = {x - pos[0], y - pos[1], z - pos[2]};
                float d2plane = dif[0] * normal[0] + dif[1] * normal[1] + dif[2] * normal[2];
                if (fabsf(d2plane) > grid_step * 0.5) continue;
                float d2axis[3] = {dif[0] - normal[0] * d2plane, dif[1] - normal[1] * d2plane, dif[2] - normal[2] * d2plane};
                // float d2 = d2axis[0] * d2axis[0] + d2axis[1] * d2axis[1] + d2axis[2] * d2axis[2];
                // // if (d2 > 0.001) continue;
                // if (expf(d2 / max_scale / max_scale / -2) < cutoff_v) continue;

                float dx = fabsf(d2axis[0] * dir_x[0] + d2axis[1] * dir_x[1] + d2axis[2] * dir_x[2]) / scale[0];
                float dy = fabsf(d2axis[0] * dir_y[0] + d2axis[1] * dir_y[1] + d2axis[2] * dir_y[2]) / scale[1];
                float d2 = dx * dx + dy * dy;
                float alpha = expf(d2 / -2) * opac;
                if (alpha < cutoff) continue;

                // write grid
                int grid_cord[3] = {(x - pmin[0]) / grid_step, (y - pmin[1]) / grid_step, (z - pmin[2]) / grid_step};
                if (grid_cord[0] > gdim[0] - 1 || grid_cord[1] > gdim[1] - 1 || grid_cord[2] > gdim[2] - 1 ||
                    grid_cord[0] < 0 || grid_cord[1] < 0 || grid_cord[2] < 0) return;
                int grid_idx = grid_cord[0] * gdim[2] * gdim[1] + grid_cord[1] * gdim[2] + grid_cord[2];
                
                // atomicCAS_f32(&grid[grid_idx], 0, 1);
                atomicAdd(&grid[grid_idx], alpha);
            }
        }
    }

    // int grid_cord[3] = {(pos[0] - pmin[0]) / grid_step, (pos[1] - pmin[1]) / grid_step, (pos[2] - pmin[2]) / grid_step};
    // int grid_idx = grid_cord[0] * gdim[2] * gdim[1] + grid_cord[1] * gdim[2] + grid_cord[2];
    // atomicAdd(&grid[grid_idx], 1);


    // for (int x)



}

void gaussians2occgrid_cuda(
    int* n_point,
    float* pos_min,
    float* pos_max,
    float* grid_len,
    int* grid_dim,
    float* grid,
    float* pos,
    float* rot,
    float* scale,
    float* opac,
    float* cutoff
) {
    const int n_thread = 512;
    int n_block = *n_point / n_thread + 1;
    int *n_point_cuda;
    cudaMalloc((void**) &n_point_cuda, 4);
    cudaMemcpy(n_point_cuda, n_point, 4, cudaMemcpyHostToDevice);

    gaussian2occgrid_kernel<<<n_block, n_thread>>>(
        n_point_cuda,
        pos_min, pos_max,
        grid_len, grid_dim, grid,
        pos, rot, scale, opac,
        cutoff
    );
}

__global__ void tsdf_fusion_kernel(
    int* n_point,
    float* pos_min,
    float* pos_max,
    float* grid_len,
    int* grid_dim,
    float* pos_in,
    float* normal_in,
    float* center_in,
    float* grid
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *n_point) return;
    float pos[3] = {pos_in[idx * 3 + 0], pos_in[idx * 3 + 1], pos_in[idx * 3 + 2]},
          normal[3] = {normal_in[idx * 3 + 0], normal_in[idx * 3 + 1], normal_in[idx * 3 + 2]},
          pmin[3] = {pos_min[0], pos_min[1], pos_min[2]}, pmax[3] = {pos_max[0], pos_max[1], pos_max[2]};
    int gdim[3] = {grid_dim[0], grid_dim[1], grid_dim[2]};
    // float mod = sqrtf(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    float viewDir[3] = {pos[0] - center_in[0], pos[1] - center_in[1], pos[2] - center_in[2]};
    float* truncate_dir = viewDir;
    float grid_step = *grid_len;
    float truncate = 1 * grid_step, step = 0.1 * grid_step;
    for (float i = -truncate; i <= truncate; i += step) {
        float pos_cur[3] = {pos[0] + i * truncate_dir[0], pos[1] + i * truncate_dir[1], pos[2] + i * truncate_dir[2]};
        int grid_cord[3] = {(pos_cur[0] - pmin[0]) / grid_step, (pos_cur[1] - pmin[1]) / grid_step, (pos_cur[2] - pmin[2]) / grid_step};
        int grid_idx = grid_cord[0] * gdim[2] * gdim[1] + grid_cord[1] * gdim[2] + grid_cord[2];
        if (grid_cord[0] > gdim[0] || grid_cord[1] > gdim[1] || grid_cord[2] > gdim[2])
            printf("%d, %d, %d\n", grid_cord[0], grid_cord[1], grid_cord[2]);
        atomicAdd(&grid[grid_idx * 2 + 0], i);
        atomicAdd(&grid[grid_idx * 2 + 1], 1);
    }



} 

void tsdf_fusion_cuda(
    int* n_point,
    float* pos_min,
    float* pos_max,
    float* grid_len,
    int* grid_dim,
    float* pos,
    float* normal,
    float* center,
    float* grid
) {
    const int n_thread = 512;
    int n_block = *n_point / n_thread + 1;
    int *n_point_cuda;
    cudaMalloc((void**) &n_point_cuda, 4);
    cudaMemcpy(n_point_cuda, n_point, 4, cudaMemcpyHostToDevice);

    tsdf_fusion_kernel<<<n_block, n_thread>>>(
        n_point_cuda,
        pos_min, pos_max,
        grid_len, grid_dim,
        pos, normal, center,
        grid
    );
}