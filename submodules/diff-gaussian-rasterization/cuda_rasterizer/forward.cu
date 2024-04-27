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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	// float3 t = transformPoint4x3(mean, viewmatrix);
	float3 t = mean;

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	const float l = sqrtf(t.x * t.x + t.y * t.y + t.z * t.z);
	// float Svp = (focal_x + focal_y) / 2;
	// printf("%.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n", mean.x, mean.y, mean.z, t.x, t.y, t.z);

	// *J = glm::mat3(
	// 	1 / t.z, 0.0f, -t.x / (t.z * t.z),
	// 	0.0f, 1 / t.z, -t.y / (t.z * t.z),
	// 	t.x / l, t.y / l, t.z / l);

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;

	// scrnAxes
	// float4 tempAxes = {scrnAxes.x * T[0][0] + scrnAxes.y * T[1][0],
	//                    scrnAxes.x * T[0][1] + scrnAxes.y * T[1][1],
	// 				   scrnAxes.z * T[0][0] + scrnAxes.w * T[1][0],
	//                    scrnAxes.z * T[0][1] + scrnAxes.w * T[1][1]};
	// scrnAxes = tempAxes;

	// glm::mat3 Jinv = inverse3x3(J);

	// glm::mat4x3 res = {
	// 	cov[0][0], cov[0][1], cov[1][1], // cov
	// 	Jinv[0][0], Jinv[0][1], Jinv[0][2],
	// 	Jinv[1][0], Jinv[1][1], Jinv[1][2],
	// 	Jinv[2][0], Jinv[2][1], Jinv[2][2]
	// };
	// return res;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__device__ glm::mat3 quaternion2rotmat(const glm::vec4 q) {
	// Normalize quaternion to get valid rotation
	// glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);
	return R;
}


// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::mat3 R, float* cov3D, bool surface)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * surface ? 0 : scale.z;
	// S[2][2] = mod * scale.z;

	// // Normalize quaternion to get valid rotation
	// glm::vec4 q = rot;// / glm::length(rot);
	// float r = q.x;
	// float x = q.y;
	// float y = q.z;
	// float z = q.w;

	// // Compute rotation matrix from quaternion
	// glm::mat3 R = glm::mat3(
	// 	1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
	// 	2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
	// 	2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	// );

	// // Store last col of R as normal
	// normal[0] = R[2][0];
	// normal[1] = R[2][1];
	// normal[2] = R[2][2];

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	// const float* cutoff,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* prcppoint,
	const float* patchbbox,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float* normal,
	float4* conic_opacity,
	// float4* cutOff,
	float* Jinv,
	float* viewCos,
	int* pid,
	float3* pview,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	float* config)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;


	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);

	// Perform near culling, quit if outside.
	float2 point_image = { ndc2Pix(p_proj.x, W, prcppoint[0]), ndc2Pix(p_proj.y, H, prcppoint[1]) };
	if (!in_frustum(p_view, p_proj, point_image, patchbbox, prefiltered)) return;

	glm::mat3 R = quaternion2rotmat(rotations[idx]);

	bool surface = config[0] > 0, pix_depth = config[2] > 0;// apply_cutoff = config[4] > 0, update_cutoff = config[5] > 0;
	float3 n_view;

	if (surface) {
		// compute camera normal from rotation matrix
		// float3 n_world = {R[0][2], R[1][2], R[2][2]};
		// float3 wrdNormal = {R[2][0], R[2][1], R[2][2]};
		// float3 wrdNormal = {R[0][0], R[0][1], R[0][2]};
		float3 n_view   = transformVec4x3({R[0][2], R[1][2], R[2][2]}, viewmatrix);
		float3 ax0_view = transformVec4x3({R[0][0], R[1][0], R[2][0]}, viewmatrix);
		float3 ax1_view = transformVec4x3({R[0][1], R[1][1], R[2][1]}, viewmatrix);
		// printf("here\n");
		if (!front_facing(n_view, p_view, &viewCos[idx], prefiltered)) return; // cull backfacing points
		// printf("points left: %.8f, %.8f, %.8f\n", p_view.x, p_view.y, p_view.z);
		normal[idx * 3 + 0] = n_view.x;
		normal[idx * 3 + 1] = n_view.y;
		normal[idx * 3 + 2] = n_view.z;
		// normal[idx * 3 + 0] = wrdNormal.x;
		// normal[idx * 3 + 1] = wrdNormal.y;
		// normal[idx * 3 + 2] = wrdNormal.z;
		// normal[idx * 3 + 0] = p_orig.x;
		// normal[idx * 3 + 1] = p_orig.y;
		// normal[idx * 3 + 2] = p_orig.z;

		if (pix_depth) {
			// compute local homography between two planes
			float Jinv_u0_u1[10];
			bool grazing = local_homo(p_view, n_view, focal_x, focal_y, ax0_view, ax1_view, Jinv_u0_u1);
			if (grazing) return; // cull grazing-faced points
			for (int i = 0; i < 10; i++) Jinv[idx * 10 + i] = Jinv_u0_u1[i];
		}
	}


	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters.
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, R, cov3Ds + idx * 6, surface);
		cov3D = cov3Ds + idx * 6;
	}


	// float4 scrnAxes = {R[0][0], R[1][0], R[0][1], R[1][1]};
	// Compute 2D screen-space covariance matrix
	// glm::mat3 J;
	float3 cov = computeCov2D(p_view, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// if (surface && pix_depth) {
	// 	bool cond = prepare_pixDepth(J, n_view, p_view, &Jinv[idx * 9]);
	// 	if (!cond) return;
	// }

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f) return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	// if (min(lambda1, lambda2) == 0 || max(lambda1, lambda2) / min(lambda1, lambda2) > 100) return;
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	

	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0) return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	// pview[idx] = p_view;
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	// if (apply_cutoff) cutOff[idx] = {cutoff[idx * 4 + 0], cutoff[idx * 4 + 1], cutoff[idx * 4 + 2], cutoff[idx * 4 + 3]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	// pid[idx] = idx;
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H, 
	const float* prcppoint,
	const float* patchbbox,
	const float focal_x, const float focal_y,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ normal,
	const float* __restrict__ depth,
	const float4* __restrict__ conic_opacity,
	// const float4* __restrict__ cutOff,
	// float* cutoff,
	const float* __restrict__ Jinv,
	const int* __restrict__ pid,
	const float3* __restrict__ pview,
	float* __restrict__ final_T,
	float* __restrict__ final_D,
	float* __restrict__ final_C,
	float* __restrict__ final_V,
	uint32_t* __restrict__ n_contrib,
	float* __restrict__ final_T_cut,
	uint32_t* __restrict__ n_contrib_cut,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_normal,
	float* __restrict__ out_depth,
	float* __restrict__ out_opac,
	// float* __restrict__ out_rayVar,
	float* config)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };


	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// bool inside = pix.x >= patchbbox[1] && pix.x < patchbbox[3] && pix.y >= patchbbox[0] && pix.y < patchbbox[2];
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;
	// if (pix.x < 180 || pix.x > 220 || pix.y > 160) done = true;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_feature[CHANNELS * BLOCK_SIZE];
	__shared__ float collected_depth[BLOCK_SIZE];
	__shared__ float collected_normal[3 * BLOCK_SIZE];
	__shared__ float collected_Jinv[10 * BLOCK_SIZE];
	__shared__ int collected_pid[BLOCK_SIZE];
	// __shared__ float4 collected_cutOff[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f, test_T;
	uint32_t contributor = 0, blend_count = 0;
	uint32_t last_contributor = 0, cut_contributor = 0;
	float C[CHANNELS] = { 0 }, N[3] = {0}, D = 0, depth_first, CUT = 0;

	bool surface = config[0] > 0, per_pixel_depth = config[2] > 0, normalize_depth = config[1] > 0;
	    //  apply_cutoff = config[4] > 0, update_cutoff = config[5] > 0;
	// float decay_weight = config[7];
	const int D_buffer_size = 128;
	float D_buffer[D_buffer_size], W_buffer[D_buffer_size], depth_temp,
	      axDif_buffer[D_buffer_size * 2], pid_buffer[D_buffer_size];

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < CHANNELS; i++) collected_feature[i * BLOCK_SIZE + block.thread_rank()] = features[coll_id * CHANNELS + i];
			collected_depth[block.thread_rank()] = depth[coll_id];
			collected_pid[block.thread_rank()] = pid[coll_id];
			for (int i = 0; i < 3; i++) collected_normal[i * BLOCK_SIZE + block.thread_rank()] = normal[coll_id * 3 + i];
			for (int i = 0; i < 10; i++) collected_Jinv[i * BLOCK_SIZE + block.thread_rank()] = Jinv[coll_id * 10 + i];
			// collected_cutOff[block.thread_rank()] = cutOff[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float dist = (con_o.x * d.x * d.x + con_o.z * d.y * d.y) + 2 * con_o.y * d.x * d.y;
			float power = -0.5f * dist;
			// float cut = cutoff[collected_id[j]].x;
			// float3 pview_dif = {pview[collected_id[j] * 3 + 0], pview[collected_id[j] * 3 + 1], pview[collected_id[j] * 3 + 2]};
			// float D = normal[collected_id[j] * 3 + 0]
			if (power > 0.0f) continue;


			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));

			// float dc = sqrtf((xy.x - 800) * (xy.x - 800) + (xy.y - 600) * (xy.y - 600));
			// if (dc > 50) {
			// 	printf("%.5f, %.5f, %.5f\n", xy.x, xy.y, dc);
			// 	alpha *= 0.1;
			// }

			// if (use_cutoff) {			
			// 	// soft cutoff to retain grad
			// 	// float alpha_decay = exp(-decay_weight * fmaxf(0, dist - cut));
			// 	// if (dist > cut) alpha *= alpha_decay;
			// 	float thrsh = 0.99;
			// 	float shift = log(thrsh / (1 - thrsh)) / decay_weight;
			// 	float alpha_decay = 1 - 1 / (1 + exp(-decay_weight * fminf(shift, fmaxf(-shift, dist - cut))));
			// 	// if (fabsf(dist - cut) < shift) {
			// 	if (dist > cut - shift) {
			// 		alpha *= alpha_decay;
			// 		// printf("%.7f, %.7f\n", shift, alpha_decay);
			// 	}
			// 	if (alpha_decay < 1.0f / 255.0f) continue;

			// }



			if (alpha < 1.0f / 255.0f) continue;
			// if (power / -0.5f > cutoff[collected_id[j]]) continue;

			test_T = T * (1 - alpha);
			if (test_T < 0.0001f) {
				done = true;
				continue;
			}
			float w = alpha * T;			
			
			depth_temp = collected_depth[j];
			if (surface && per_pixel_depth) {
				float Jinv_temp[10];
				for (int ch = 0; ch < 10; ch++) Jinv_temp[ch] = collected_Jinv[ch * BLOCK_SIZE + j];
				
				// printf("%.8f, %.8f\n", J_inv[2], J_inv[3]);
				float3 pos_dif = depth_differencing(d, Jinv_temp);
				// float depth_dif = get_pixDepth(d, J_inv, focal_x, focal_y);
				// if (depth_dif < 0.01) return;
				// printf("%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n", depth_temp, depth_dif, w, alpha, T,
				// collected_normal[0 * BLOCK_SIZE + j], collected_normal[1 * BLOCK_SIZE + j], collected_normal[2 * BLOCK_SIZE + j]);

				// float4 cutOff_thrsh = collected_cutOff[j];
				// if (apply_cutoff && 
				//     	(pos_dif.x < cutOff_thrsh.x || pos_dif.x > cutOff_thrsh.y || 
				//     	 pos_dif.y < cutOff_thrsh.z || pos_dif.y > cutOff_thrsh.w)) {
				// 	// printf("\n%.5f, %.5f, %.5f, %.5f, %.5f, %.5f", 
				// 	// cutOff_thrsh.x, cutOff_thrsh.y, cutOff_thrsh.z, cutOff_thrsh.w, pos_dif.x, pos_dif.y);
				// 	continue;
				// }


				depth_temp -= pos_dif.z;
				// depth_temp -= depth_dif * W;
				if (blend_count < D_buffer_size) {
					axDif_buffer[blend_count * 2 + 0] = -pos_dif.x;
					axDif_buffer[blend_count * 2 + 1] = -pos_dif.y;
				}
			}
			D += depth_temp * w;
			// if (blend_count == 0) depth_first = depth_temp;
			
			if (blend_count < D_buffer_size && T > 0.1) {
				D_buffer[blend_count] = depth_temp;
				W_buffer[blend_count] = w;
				pid_buffer[blend_count] = collected_pid[j];
				blend_count++;
			}


			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++) C[ch] += collected_feature[ch * BLOCK_SIZE + j] * w;
			
			if (surface) for (int ch = 0; ch < 3; ch++) N[ch] += collected_normal[ch * BLOCK_SIZE + j] * w;
			


			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;

			// if (blend_count == 1) {
			// 	done = true;
			// 	continue;
			// }

		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		T = fminf(1 - 0.000001, T);


		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++) {
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		}
		out_normal[0 * H * W + pix_id] = surface ? N[0] : 0;
		out_normal[1 * H * W + pix_id] = surface ? N[1] : 0;
		out_normal[2 * H * W + pix_id] = surface ? N[2] : 0;
		out_depth[pix_id] = normalize_depth ? D / (1 - T) : D + T * 10;
		out_opac[pix_id] = 1 - T;
		if (normalize_depth) final_D[pix_id] = D;

		// // Calculate depth variance along a ray
		// float D_var = 0, D_mean = D / (1 - T), D_dif_mean, D_dif_mid, D_max = 5;
		// float D_mid = D_buffer[(int)(min(D_buffer_size, blend_count) / 2)];
		// for (int i = 0; i < min(D_buffer_size, blend_count); i++) {
		// 	D_dif_mean = (D_buffer[i] - D_mean) / D_max;
		// 	D_var += D_dif_mean * D_dif_mean * W_buffer[i];
		// 	// update cutoff value
		// 	if (update_cutoff) {
		// 		float cutoff_thrsh = config[5];
		// 		D_dif_mid = (D_mid - D_buffer[i]);
		// 		if (fabsf(D_dif_mid) > cutoff_thrsh) {
		// 			// printf("%.5f ", D_dif_mid);
		// 			out_opac[pix_id] = 0;
		// 			return;
		// 			// project to plane
		// 			// cut axis
		// 			// write use pid
		// 			// printf("updating cutoffs\n");
		// 			float dx = axDif_buffer[i * 2 + 0];
		// 			float dy = axDif_buffer[i * 2 + 1];
		// 			int pid = pid_buffer[i];
		// 			if (dx < 0) {
		// 				atomicMax_f32(&cutoff[pid * 4 + 0], dx);
		// 				// float old = cutoff[pid * 4 + 0];
		// 				// while (old < dx) {
		// 				// 	old = atomicCAS_f32(&cutoff[pid * 4 + 0], old, dx);
		// 				// }
		// 			}
		// 			else {
		// 				atomicMin_f32(&cutoff[pid * 4 + 1], dx);
		// 				// float old = cutoff[pid * 4 + 1];
		// 				// while (old > dx) {
		// 				// 	old = atomicCAS_f32(&cutoff[pid * 4 + 1], old, dx);
		// 				// }
		// 			}
		// 			if (dy < 0) {
		// 				atomicMax_f32(&cutoff[pid * 4 + 2], dy);
		// 				// float old = cutoff[pid * 4 + 0];
		// 				// while (old < dx) {
		// 				// 	old = atomicCAS_f32(&cutoff[pid * 4 + 0], old, dx);
		// 				// }
		// 			}
		// 			else {
		// 				atomicMin_f32(&cutoff[pid * 4 + 3], dy);
		// 				// float old = cutoff[pid * 4 + 1];
		// 				// while (old > dx) {
		// 				// 	old = atomicCAS_f32(&cutoff[pid * 4 + 1], old, dx);
		// 				// }
		// 			}
		// 		}
		// 	}
		// }
	
	}
	// else {
	// 	out_depth[pix_id] = 0; // write 0 to depth, use it to compute mask
	// }
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H, 
	const float* prcppoint,
	const float* patchbbox,
	const float focal_x, const float focal_y,
	const float2* means2D,
	const float* colors,
	const float* normal,
	const float* depth,
	const float4* conic_opacity,
	// const float4* cutOff,
	// float* cutoff, // use for hard modify
	const float* Jinv,
	const int* pid,
	const float3* pview,
	float* final_T,
	float* final_D,
	float* final_C,
	float* final_V,
	uint32_t* n_contrib,
	float* final_T_cut,
	uint32_t* n_contrib_cut,
	const float* bg_color,
	float* out_color,
	float* out_normal,
	float* out_depth,
	float* out_opac,
	// float* out_rayVar,
	float* config)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H, 
		prcppoint,
		patchbbox,
		focal_x, focal_y,
		means2D,
		colors,
		normal,
		depth,
		conic_opacity,
		// cutOff,
		// cutoff,
		Jinv,
		pid,
		pview,
		final_T,
		final_D,
		final_C,
		final_V,
		n_contrib,
		final_T_cut,
		n_contrib_cut,
		bg_color,
		out_color, out_normal, out_depth, out_opac, //out_rayVar, 
		config);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	// const float* cutoff,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* prcppoint,
	const float* patchbbox,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float* normal,
	float4* conic_opacity,
	// float4* cutOff,
	float* Jinv,
	float* viewCos,
	int* pid,
	float3* pview,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	float* config)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		// cutoff,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		prcppoint,
		patchbbox,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		normal,
		conic_opacity,
		// cutOff,
		Jinv,
		viewCos,
		pid,
		pview,
		grid,
		tiles_touched,
		prefiltered,
		config
		);
}