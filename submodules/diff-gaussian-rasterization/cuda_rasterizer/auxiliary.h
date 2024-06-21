/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"
#include <glm/glm.hpp>

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S, float prcp)
{
	// return ((v + 1.0) * S - 1.0) * 0.5;
	return ((v + 1.0) * S - 1.0) * 0.5; //+ S * (prcp - 0.5);
}

__forceinline__ __device__ float pix2Ndc(float v, int S, float prcp)
{
	return ((v - S * (prcp - 0.5)) * 2.0 + 1.0); /// S - 1.0;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(
	float3& p_view,
	float3& p_proj, 
	float2& p_pix, 
	const float* patchbbox,
	bool prefiltered)
{

	// if ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3))
	// 	printf("p_proj out of frustum! %.8f, %.8f, %.8f\n", p_proj.x, p_proj.y, p_proj.z);
	// float expand = 1.1;
	// if (p_view.z < 0 || ((p_proj.x < -expand || p_proj.x > expand || p_proj.y < -expand || p_proj.y > expand)))
	float x0 = patchbbox[1], y0 = patchbbox[0], x1 = patchbbox[3], y1 = patchbbox[2];
	float w = x1 - x0, h = y1 - y0;
	float expand = 0.2;
	if (p_view.z < 0 || p_pix.x < x0 - w * expand || p_pix.x >= x1 + w * expand || p_pix.y < y0 - h * expand || p_pix.y >= y1 + h * expand)
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

__forceinline__ __device__ bool front_facing(
	float3& n_view,
	float3& p_view,
	float* viewCos,
	bool prefiltered)
{
	float dot = p_view.x * n_view.x + p_view.y * n_view.y + p_view.z * n_view.z;
	// float z = n_view.z;
	bool cond = (dot > -0.01);

	// float sin = dot / sqrtf(p_view.x * p_view.x + p_view.y * p_view.y + p_view.z * p_view.z);
	// if (dot < 0 && n_view.z >= 0)

	// cond = (dot >= 0 || z < 0 || p_view.y < 0 || p_view.z > 0);
	// cond = dot < 0 && z >=0;
	// cond = p_view.z < 0;
	// cond = p_view.z < 0;
	// cond = (dot >= 0 || sin > 0.95);
	// cond = dot >= 0;

	if (cond)
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}

		// printf("back facing normal: %.8f, %.8f, %.8f\n", camNormal.x, camNormal.y, camNormal.z);
		return false;
	}

	// printf("front facing normal: %.8f, %.8f, %.8f\n", camNormal.x, camNormal.y, camNormal.z);
	*viewCos = dot;
	return true;
}

__forceinline__ __device__ glm::mat3 inverse3x3(
	glm::mat3 M)
{
	glm::mat3 A = {
         M[1][1] * M[2][2] - M[1][2] * M[2][1], // 00
        -M[0][1] * M[2][2] + M[0][2] * M[2][1], // 10
         M[0][1] * M[1][2] - M[0][2] * M[1][1], // 20
        -M[1][0] * M[2][2] + M[2][0] * M[1][2], // 01
         M[0][0] * M[2][2] - M[0][2] * M[2][0], // 11
        -M[0][0] * M[1][2] + M[0][2] * M[1][0], // 21
         M[1][0] * M[2][1] - M[1][1] * M[2][0], // 02
        -M[0][0] * M[2][1] + M[0][1] * M[2][0], // 12
         M[0][0] * M[1][1] - M[0][1] * M[1][0], // 22
	};
    float det = M[0][0] * A[0][0] + M[0][1] * A[1][0] + M[0][2] * A[2][0];
    glm::mat3 Minv = A / det;

	// glm::mat3 mult = M * Minv;
	// printf("\n%.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f\n%.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f\n%.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f\n", 
	// M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2], M[2][0], M[2][1], M[2][2], 
	// Minv[0][0], Minv[0][1], Minv[0][2], Minv[1][0], Minv[1][1], Minv[1][2], Minv[2][0], Minv[2][1], Minv[2][2],
	// mult[0][0], mult[0][1], mult[0][2], mult[1][0], mult[1][1], mult[1][2], mult[2][0], mult[2][1], mult[2][2]);

    return Minv;
}

__forceinline__ __device__ float normalize(float* v) {
    float mod = fmaxf(sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), 0.00000001);
    v[0] /= mod;
    v[1] /= mod;
    v[2] /= mod;
    return mod;
}

// __forceinline__ __device__ bool prepare_pixDepth(
// 	glm::mat3 J, float3 n_view, float3 p_view, float* data_pixDepth)
// {
// 	glm::mat3 Jinv = inverse3x3(J);
// 	float3 p_ray = {p_view.x / p_view.z, p_view.y / p_view.z, 
// 	                sqrtf(p_view.x * p_view.x + p_view.y * p_view.y + p_view.z * p_view.z)};
// 	float3 p_view_ = {p_view.x + 0.001 * n_view.x, p_view.y + 0.001 * n_view.y, p_view.z + 0.001 * n_view.z};
// 	float3 p_ray_ = {p_view_.x / p_view_.z, p_view_.y / p_view_.z, 
// 	                 sqrtf(p_view_.x * p_view_.x + p_view_.y * p_view_.y + p_view_.z * p_view_.z)};
// 	float n_ray[3] = {p_ray_.x - p_ray.x, p_ray_.y - p_ray.y, p_ray_.z - p_ray.z};

// 	// if (n_ray[2] > -0.05) return false;

// 	normalize(n_ray);
	
// 	// float data_pixDepth[9] = {p_ray.x, p_ray.y, p_ray.z, n_ray.x, n_ray.y, n_ray.z, Jinv[2][0], Jinv[2][1], Jinv[2][2]};
// 	data_pixDepth[0] = p_ray.x;
// 	data_pixDepth[1] = p_ray.y;
// 	data_pixDepth[2] = p_ray.z;
// 	data_pixDepth[3] = n_ray[0];
// 	data_pixDepth[4] = n_ray[1];
// 	data_pixDepth[5] = n_ray[2];
// 	// data_pixDepth[3] = n_view.x;
// 	// data_pixDepth[4] = n_view.y;
// 	// data_pixDepth[5] = n_view.z;
// 	data_pixDepth[6] = Jinv[2][0];
// 	data_pixDepth[7] = Jinv[2][1];
// 	data_pixDepth[8] = Jinv[2][2];
// 	return true;
	
// }

// __forceinline__ __device__ float get_pixDepth(
// 	float2 d_scrn, float* data_pixDepth, const float fx, const float fy) {
// 	// float2 p_proj = {pix2Ndc(p_scrn.x, W, prcp[0]), pix2Ndc(p_scrn.y, H, prcp[1])};
// 	// printf("%.5f, %.5f, %.5f, %.5f\n", p_proj.x, p_proj.y, data_pixDepth[0], data_pixDepth[1]);
// 	// float Svp = (fx + fy) / 2;
// 	float3 d_ray = {d_scrn.x / fx, d_scrn.y / fy, 0};
// 	// float3 p_ray = {data_pixDepth[0], data_pixDepth[1], data_pixDepth[2]};
// 	float3 n_ray = {data_pixDepth[3], data_pixDepth[4], data_pixDepth[5]};
// 	d_ray.z = -(d_ray.x * n_ray.x + d_ray.y * n_ray.y) / n_ray.z;
// 	float3 Jinv_r2 = {data_pixDepth[6], data_pixDepth[7], data_pixDepth[8]};
// 	float d_view_z = (d_ray.x * Jinv_r2.x + d_ray.y * Jinv_r2.y + d_ray.z * Jinv_r2.z);
// 	// printf("%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n", d_ray.x, d_ray.y, d_ray.z, Jinv_r2.x, Jinv_r2.y, Jinv_r2.z, d_view_z);
// 	return d_view_z * 10;
// }

__forceinline__ __device__ bool local_homo(
	float3 p_view, float3 n_view, float fx, float fy, float3 ax0, float3 ax1, float* res)
{
	// project screen unit to tangent plane to calculate J
    // view direction of screen unit
	float2 p_prj = {p_view.x / p_view.z, p_view.y / p_view.z};
	float S_fix = 1000, Svp = (fx + fy) / 2;
	// p_prj.x += W / Svp / 2 * (px - 0.5);
	// p_prj.y += H / Svp / 2 * (py - 0.5);
	// printf("%.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n", p_prj.x, p_prj.y, W / Svp * (px - 0.5), H / Svp * (py - 0.5), px, py);
    float dir_x0[3], dir_x1[3];
    dir_x0[0] = p_prj.x + 1 / S_fix;
    dir_x0[1] = p_prj.y;
    dir_x0[2] = 1;
    float dir_x0_mod = normalize(dir_x0);
    dir_x1[0] = p_prj.x;
    dir_x1[1] = p_prj.y + 1 / S_fix;
    dir_x1[2] = 1;
    float dir_x1_mod = normalize(dir_x1);

	// n_view.x = 0;
	// n_view.y = 0;
	// n_view.z = -1;
    // cutoff extreme projection angle
    // extreme case results in very long 'ellipse' in wrong direction,
    // it becomes more severe and frequent when observe from faraway.
    float prj_x0, prj_x1, thrsh_prj = 0.01;
    prj_x0 = dir_x0[0] * n_view.x + dir_x0[1] * n_view.y + dir_x0[2] * n_view.z;
    prj_x1 = dir_x1[0] * n_view.x + dir_x1[1] * n_view.y + dir_x1[2] * n_view.z;
    bool cond_prj = (fabsf(prj_x0 / dir_x0_mod) < thrsh_prj) || (fabsf(prj_x1 / dir_x1_mod) < thrsh_prj);
    if (cond_prj) return true;
    

    // projected screen unit
    float t_temp, t_x0, t_x1, xu0[3], xu1[3], u0[3], u1[3], xu0_mod;
    t_temp = p_view.x * n_view.x + p_view.y * n_view.y + p_view.z * n_view.z;
    t_x0 = t_temp / prj_x0;
    t_x1 = t_temp / prj_x1;
	xu0[0] = dir_x0[0] * t_x0 - p_view.x;
	xu0[1] = dir_x0[1] * t_x0 - p_view.y;
	xu0[2] = dir_x0[2] * t_x0 - p_view.z;
	xu1[0] = dir_x1[0] * t_x1 - p_view.x;
	xu1[1] = dir_x1[1] * t_x1 - p_view.y;
	xu1[2] = dir_x1[2] * t_x1 - p_view.z;

	// printf("%.5f, %.5f,     %.5f, %.5f, %.5f,     %.5f, %.5f, %.5f\n",
	// xu1[0] * n_view.x + xu1[1] * n_view.y + xu1[2] * n_view.z,
	// xu0[0] * n_view.x + xu0[1] * n_view.y + xu0[2] * n_view.z,
	// xu1[0], xu1[1], xu1[2], xu0[0], xu0[1], xu0[2]);
	// printf("%.5f, %.5f, %.5f,     %.5f, %.5f, %.5f\n",
	// p_prj.x, p_prj.y, p_prj.z,
	// p_view.x / p_view.z, p_view.y / p_view.z, 1);

	// printf("%.5f, ", xu0_mod);


    // tangent space unit
		// original method in Surface Splatting:
		xu0_mod = fmaxf(sqrtf(xu0[0] * xu0[0] + xu0[1] * xu0[1] + xu0[2] * xu0[2]), 0.00000001);
		u0[0] = xu0[0] / xu0_mod;
		u0[1] = xu0[1] / xu0_mod;
		u0[2] = xu0[2] / xu0_mod;
		u1[0] = u0[1] * n_view.z - u0[2] * n_view.y;
		u1[1] = u0[2] * n_view.x - u0[0] * n_view.z;
		u1[2] = u0[0] * n_view.y - u0[1] * n_view.x;

		// using R inv viewspace as local tangent coordinates
		u0[0] = ax0.x;
		u0[1] = ax0.y;
		u0[2] = ax0.z;
		u1[0] = ax1.x;
		u1[1] = ax1.y;
		u1[2] = ax1.z;


	float J_inv[4];
    J_inv[0] = xu0[0] * u0[0] + xu0[1] * u0[1] + xu0[2] * u0[2];
    J_inv[1] = xu1[0] * u0[0] + xu1[1] * u0[1] + xu1[2] * u0[2];
    J_inv[2] = xu0[0] * u1[0] + xu0[1] * u1[1] + xu0[2] * u1[2];
    J_inv[3] = xu1[0] * u1[0] + xu1[1] * u1[1] + xu1[2] * u1[2];

    J_inv[0] /= (Svp / S_fix); // scale & scale back
    J_inv[1] /= (Svp / S_fix);
    J_inv[2] /= (Svp / S_fix);
    J_inv[3] /= (Svp / S_fix);
	
	res[0] = J_inv[0];
	res[1] = J_inv[1];
	res[2] = J_inv[2];
	res[3] = J_inv[3];
	for (int i = 0; i < 3; i++) {
		res[4 + i] = u0[i];
		res[7 + i] = u1[i];
	}
	return false;

	
}

__forceinline__ __device__ float3 depth_differencing(float2 pix_dif, float* Jinv_u0_u1) {
	float dif_u[2] = {pix_dif.x * Jinv_u0_u1[0] + pix_dif.y * Jinv_u0_u1[1], 
	                  pix_dif.x * Jinv_u0_u1[2] + pix_dif.y * Jinv_u0_u1[3]};
	float3 pos_dif = {dif_u[0] * Jinv_u0_u1[4] + dif_u[1] * Jinv_u0_u1[7],
					  dif_u[0] * Jinv_u0_u1[5] + dif_u[1] * Jinv_u0_u1[8],
					  dif_u[0] * Jinv_u0_u1[6] + dif_u[1] * Jinv_u0_u1[9]};
	return pos_dif;
}

__forceinline__ __device__ float atomicCAS_f32(float *p, float cmp, float val) {
    int res = atomicCAS((int*)p, *(int*)&cmp, *(int*)&val);
    // printf("%.5f\n", *(float*)&res);
    return *(float*)&res;
}

__forceinline__ __device__ float atomicMax_f32(float* p, float val) {
	float old = *p;
	while (old < val) {
		old = atomicCAS_f32(p, old, val);
	}
}

__forceinline__ __device__ float atomicMin_f32(float* p, float val) {
	float old = *p;
	while (old > val) {
		old = atomicCAS_f32(p, old, val);
	}
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif