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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(
	int idx, 
	int deg, 
	int max_coeffs, 
	const glm::vec3* means, 
	glm::vec3 campos, 
	const float* shs, 
	const bool* clamped, 
	const glm::vec3* dL_dcolor, 
	glm::vec3* dL_dmeans, 
	glm::vec3* dL_dshs,
	bool lrn_cam,
	float* dL_dcampos)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	if (lrn_cam) {
		// grads for campos from viewdir used in sh
		atomicAdd(&dL_dcampos[0], -dL_dmean.x);
		atomicAdd(&dL_dcampos[1], -dL_dmean.y);
		atomicAdd(&dL_dcampos[2], -dL_dmean.z);
	}

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(
	int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov,
	float* dL_dviewmat,
	float* config)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	float J0 = h_x / t.z, J1 = -(h_x * t.x) / (t.z * t.z), J2 = h_y / t.z, J3 = -(h_y * t.y) / (t.z * t.z);
	glm::mat3 J = glm::mat3(J0, 0.0f, J1,
							0.0f, J2, J3,
							0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;


	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	bool lrn_cam = config[3] > 0;
	if (lrn_cam) {
		// grads from loss to camera EXTRINSICS in W
		// float dL_dW[16] = {
		// 	dL_dT00 * J0, dL_dT01 * J0, dL_dT02 * J0, 0,
		// 	dL_dT10 * J2, dL_dT11 * J2, dL_dT12 * J2, 0,
		// 	dL_dT00 * J1 + dL_dT10 * J3, dL_dT01 * J1 + dL_dT11 * J3, dL_dT02 * J1 + dL_dT12 * J3, 0,
		// 	0, 0, 0, 0
		// };	
		float dL_dW[16] = {
			dL_dT00 * J0, dL_dT10 * J2, dL_dT00 * J1 + dL_dT10 * J3, 0,
			dL_dT01 * J0, dL_dT11 * J2, dL_dT01 * J1 + dL_dT11 * J3, 0,
			dL_dT02 * J0, dL_dT12 * J2, dL_dT02 * J1 + dL_dT12 * J3, 0,
			0, 0, 0, 0
		};

		for (int i = 0; i < 16; i++) atomicAdd(&dL_dviewmat[i], dL_dW[i]);
		
		// grads from loss to camera INTRINSICS in J
		float2 dL_dfocal = {dL_dJ00 * tz - dL_dJ02 * t.x * tz2, dL_dJ11 * tz - dL_dJ12 * t.y * tz2};
	}



	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(
	int idx, 
	const glm::vec3 scale, 
	float mod, 
	const glm::vec4 rot, 
	const float* dL_dcov3Ds, 
	glm::vec3* dL_dscales, 
	glm::vec4* dL_drots, 
	const glm::vec3* dL_dnormal,
	const float* view,
	bool surface, bool lrn_cam,
	float* dL_dviewmat)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;


	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = surface ? 0 : glm::dot(Rt[2], dL_dMt[2]);
	// dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	glm::mat3 dL_dRt = dL_dMt;
	dL_dRt[0] *= s.x;
	dL_dRt[1] *= s.y;
	dL_dRt[2] *= s.z;

	// Add loss gradient from normal output w.r.t. matrix R
	glm::vec3 dL_dcamN = dL_dnormal[idx];
	glm::vec3 dL_dwrdN = {
		dL_dcamN.x * view[0] + dL_dcamN.y * view[1] + dL_dcamN.z * view[2],
		dL_dcamN.x * view[4] + dL_dcamN.y * view[5] + dL_dcamN.z * view[6],
		dL_dcamN.x * view[8] + dL_dcamN.y * view[9] + dL_dcamN.z * view[10]
	};
	dL_dRt[2][0] += dL_dwrdN.x;
	dL_dRt[2][1] += dL_dwrdN.y;
	dL_dRt[2][2] += dL_dwrdN.z;

	if (lrn_cam) {
		// grads from normal loss to view matrix
		float3 wrdN = {R[0][2], R[1][2], R[2][2]}; 
		float dL_dviewmat_normal[16] = {
			dL_dcamN.x * wrdN.x, dL_dcamN.y * wrdN.x, dL_dcamN.z * wrdN.x, 0,
			dL_dcamN.x * wrdN.y, dL_dcamN.y * wrdN.y, dL_dcamN.z * wrdN.y, 0,
			dL_dcamN.x * wrdN.z, dL_dcamN.y * wrdN.z, dL_dcamN.z * wrdN.z, 0,
			0, 0, 0, 0
		};
		for (int i = 0; i < 16; i++) atomicAdd(&dL_dviewmat[i], dL_dviewmat_normal[i]);
	}



	// printf("\n%.5f, %.5f, %.5f, \n%.5f, %.5f, %.5f, \n%.5f, %.5f, %.5f\n", 
	// dL_dRt[0][0], dL_dRt[0][1], dL_dRt[0][2], dL_dRt[1][0], dL_dRt[1][1], dL_dRt[1][2], dL_dRt[2][0], dL_dRt[2][1], dL_dRt[2][2]);
	// dL_dM[0][2], dL_dM[1][2], dL_dM[2][2], dL_dRcol2.x, dL_dRcol2.y, dL_dRcol2.z, s.x, s.y, s.z);

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dRt[0][1] - dL_dRt[1][0]) + 2 * y * (dL_dRt[2][0] - dL_dRt[0][2]) + 2 * x * (dL_dRt[1][2] - dL_dRt[2][1]);
	dL_dq.y = 2 * y * (dL_dRt[1][0] + dL_dRt[0][1]) + 2 * z * (dL_dRt[2][0] + dL_dRt[0][2]) + 2 * r * (dL_dRt[1][2] - dL_dRt[2][1]) - 4 * x * (dL_dRt[2][2] + dL_dRt[1][1]);
	dL_dq.z = 2 * x * (dL_dRt[1][0] + dL_dRt[0][1]) + 2 * r * (dL_dRt[2][0] - dL_dRt[0][2]) + 2 * z * (dL_dRt[1][2] + dL_dRt[2][1]) - 4 * y * (dL_dRt[2][2] + dL_dRt[0][0]);
	dL_dq.w = 2 * r * (dL_dRt[0][1] - dL_dRt[1][0]) + 2 * x * (dL_dRt[2][0] + dL_dRt[0][2]) + 2 * y * (dL_dRt[1][2] + dL_dRt[2][1]) - 4 * z * (dL_dRt[1][1] + dL_dRt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* view,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dnormal,
	float* dL_ddepth,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dviewmat,
	float* dL_dprojmat,
	float* dL_dcampos,
	float* config)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);
	// float3 p_view = transformPoint4x3(m, view);

	bool surface = config[0] > 0, lrn_cam = config[3] > 0;
	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	float2 dL_dmean2d = {dL_dmean2D[idx].x, dL_dmean2D[idx].y};
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2d.x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2d.y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2d.x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2d.y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2d.x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2d.y;


	// Add position gradients from depth output
	float dL_dd = dL_ddepth[idx];
	glm::vec3 dL_dmean_fromD;
	dL_dmean_fromD.x = dL_dd * view[2];
	dL_dmean_fromD.y = dL_dd * view[6];
	dL_dmean_fromD.z = dL_dd * view[10];

	dL_dmeans[idx] += dL_dmean + dL_dmean_fromD;
	// if (dL_ddepth[idx] != 0) printf("%.10f, %.10f, %.10f\n", dL_dmean_fromD.x, dL_dmean_fromD.y, dL_dmean_fromD.z);

	if (lrn_cam) {
		// grads w.r.t. camera projection matrix
		float dL_dprojmat_proj[16] = {
			dL_dmean2d.x * m.x * m_w, dL_dmean2d.y * m.x * m_w, 0, dL_dmean2d.x * -mul1 * m.x + dL_dmean2d.y * -mul2 * m.x,
			dL_dmean2d.x * m.y * m_w, dL_dmean2d.y * m.y * m_w, 0, dL_dmean2d.x * -mul1 * m.y + dL_dmean2d.y * -mul2 * m.y,
			dL_dmean2d.x * m.z * m_w, dL_dmean2d.y * m.z * m_w, 0, dL_dmean2d.x * -mul1 * m.z + dL_dmean2d.y * -mul2 * m.z,
			dL_dmean2d.x * m_w,       dL_dmean2d.y * m_w,       0, dL_dmean2d.x * -mul1       + dL_dmean2d.y * -mul2
		};
		for (int i = 0; i < 16; i++) atomicAdd(&dL_dprojmat[i], dL_dprojmat_proj[i]);

		// grads from depth loss to view matrix
		float dL_dviewmat_depth[16] = {
			0, 0, dL_dd * m.x, 0,
			0, 0, dL_dd * m.y, 0,
			0, 0, dL_dd * m.z, 0,
			0, 0, dL_dd,       0
		};
		for (int i = 0; i < 16; i++) atomicAdd(&dL_dviewmat[i], dL_dviewmat_depth[i]);
	}

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh, lrn_cam, dL_dcampos);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot, (glm::vec3*)dL_dnormal, view, surface, lrn_cam, dL_dviewmat);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ patchbbox,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ normal,
	const float* __restrict__ depth,
	const float* __restrict__ Jinv,
	const float* __restrict__ viewCos,
	const float* __restrict__ final_Ts,
	const float* __restrict__ final_D,
	const float* __restrict__ final_C,
	const float* __restrict__ final_V,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ final_T_cut,
	const uint32_t* __restrict__ n_contrib_cut,
	const float* __restrict__ dL_dpixcolor,
	const float* __restrict__ dL_dpixnormal,
	const float* __restrict__ dL_dpixdepth,
	const float* __restrict__ dL_dpixopac,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dnormal,
	float* __restrict__ dL_ddepth,
	float* config)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	// bool inside = pix.x >= patchbbox[1] && pix.x < patchbbox[3] && pix.y >= patchbbox[0] && pix.y < patchbbox[2];

	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;
	bool surface = config[0] > 0, per_pixel_depth = config[2] > 0, normalize_depth = config[1] > 0;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_normal[3 * BLOCK_SIZE];
	__shared__ float collected_depth[BLOCK_SIZE];
	// __shared__ float4 collected_cutoff[BLOCK_SIZE];
	__shared__ float collected_Jinv[10 * BLOCK_SIZE];
	__shared__ float collected_viewCos[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	// const float T_final_cut = inside ? final_T_cut[pix_id] : 0;
	const float D_final = inside && normalize_depth ? final_D[pix_id] : 0;
	const float V_final = inside ? final_V[pix_id] : 0;

	float T = T_final;
	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;
	// const int last_contributor_cut = inside ? n_contrib_cut[pix_id] : 0;

	float accum_rec[C] = { 0 }, accum_rec_n[3] = {0}, accum_rec_d = 0, accum_rec_v = 0;
	float dL_dpixC[C], dL_dpixN[3], dL_dpixD, dL_dpixO, dL_dpixV;
	if (inside) {
		for (int i = 0; i < C; i++) dL_dpixC[i] = dL_dpixcolor[i * H * W + pix_id];
		for (int i = 0; i < 3; i++) dL_dpixN[i] = dL_dpixnormal[i * H * W + pix_id];
		dL_dpixD = dL_dpixdepth[pix_id] * 1;
		dL_dpixO = dL_dpixopac[pix_id];
		// dL_dpixV = dL_dpixrayVar[pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 }, last_normal[3] = {0}, last_depth = 0, last_var = 0;

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++) collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			for (int i = 0; i < 3; i++) collected_normal[i * BLOCK_SIZE + block.thread_rank()] = normal[coll_id * 3 + i];
			collected_depth[block.thread_rank()] = depth[coll_id];
			// if (use_cutoff) collected_cutoff[block.thread_rank()] = cutoff[coll_id];
			if (per_pixel_depth) for (int i = 0; i < 10; i++) collected_Jinv[i * BLOCK_SIZE + block.thread_rank()] = Jinv[coll_id * 10 + i];
			collected_viewCos[block.thread_rank()] = viewCos[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Skip grads for grazing points
			// if (surface && collected_viewCos[j] > -0.01) continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float dist = (con_o.x * d.x * d.x + con_o.z * d.y * d.y) + 2 * con_o.y * d.x * d.y;
			const float power = -0.5f * dist;
			if (power > 0.0f) continue;

			const float G = exp(power);
			float alpha = min(0.99f, con_o.w * G);
			
			// soft cutoff to retain grad
			// const float cut = use_cutoff ? collected_cutoff[j].x : 0;
			// float thrsh = use_cutoff ? 0.99 : 0;
			// float shift = use_cutoff ? log(thrsh / (1 - thrsh)) / decay_weight : 0;
			// float alpha_decay = use_cutoff ? 1 - 1 / (1 + exp(-decay_weight * fminf(shift, fmaxf(-shift, dist - cut)))) : 0;
			// if (use_cutoff) {
			// 	if (dist > cut - shift) alpha *= alpha_decay;
			// 	if (alpha_decay < 1.0f / 255.0f) continue;
			// }

			if (alpha < 1.0f / 255.0f) continue;
			// if (power / -0.5f > 1) continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;


			// preparation if depth differencing is required
			float Jinv_u0_u1[10] = {0};
			if (surface && per_pixel_depth) {
				for (int ch = 0; ch < 10; ch++) Jinv_u0_u1[ch] = collected_Jinv[ch * BLOCK_SIZE + j];
			}

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f, dL_dalpha_var = 0;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++) {
				const float c_cur = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c_cur;

				float dL_dchannel = dL_dpixC[ch], dL_dalpha_color = 0;

				dL_dalpha_color += (c_cur - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				// printf("%.5f %.5f\n", dchannel_dcolor, dL_dchannel);
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
				dL_dalpha += dL_dalpha_color;
			}
			// if (cut_ray_geom == 0 || contributor < last_contributor_cut) {
			if (surface) {
				for (int ch = 0; ch < 3; ch++) { // grads w.r.t. tangent normal	
					const float n_cur = collected_normal[ch * BLOCK_SIZE + j];
					// Update last normal (to be used in the next iteration)
					accum_rec_n[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_rec_n[ch];
					last_normal[ch] = n_cur;

					float dL_dchannel = dL_dpixN[ch], dL_dalpha_normal = 0;
					dL_dalpha_normal += (n_cur - accum_rec_n[ch]) * dL_dchannel;
					// Update the gradients w.r.t. normal of the Gaussian. 
					// Atomic, since this pixel is just one of potentially
					// many that were affected by this Gaussian.
					atomicAdd(&(dL_dnormal[global_id * 3 + ch]), dchannel_dcolor * dL_dchannel * 10);
					dL_dalpha += dL_dalpha_normal;
				}

			}
			// if (use_cutoff) { // grads w.r.t. ray depth variance
				// float d_cur = collected_depth[j], d_mean = D_final / (1.f - T_final), d_max = 5;
				// if (surface && per_pixel_depth) {
				// 	float depth_dif = depth_differencing(d, Jinv_u0_u1);
				// 	d_cur -= depth_dif;
				// }
				// float temp = (d_cur - d_mean) / d_max;
				// float v_cur = temp * temp; 

				// // Update last depth (to be used in the next iteration)
				// accum_rec_v = last_alpha * last_var + (1.f - last_alpha) * accum_rec_v;
				// last_var = v_cur;

				// float dL_dchannel = dL_dpixV;// dL_dalpha_var = 0;
				// // if (dL_dpixV != 0) printf("%.10f, ", dL_dpixV);
				// // return;

				// dL_dchannel /= (1.f - T_final);
				// dL_dalpha_var += dL_dpixV * V_final / (1.f - T_final) / (1.f - T_final) * -T_final / (1 - alpha) / T;	
				// dL_dalpha_var += (v_cur - accum_rec_v) * dL_dchannel;
				// dL_dalpha += dL_dalpha_var;

				// float dL_dvar = dchannel_dcolor * dL_dchannel * 1; // dvar_ddepth_i, dvar_ddepth_mean
				//float dL_ddepthi_var = dL_dvar * (2 * temp * alpha * T / (1.f - T_final));
				//float dL_ddepthmean_var = dL_dvar * (-2 * temp * alpha * T / (1.f - T_final));
			// }
			for (int ch = 0; ch < 1; ch++) { // grads w.r.t. depth
				float d_cur = collected_depth[j];
				if (surface && per_pixel_depth) {
					float3 pos_dif = depth_differencing(d, Jinv_u0_u1);
					d_cur -= pos_dif.z;
				}

				// Update last depth (to be used in the next iteration)
				accum_rec_d = last_alpha * last_depth + (1.f - last_alpha) * accum_rec_d;
				last_depth = d_cur;

				float dL_dchannel = dL_dpixD, dL_dalpha_depth = 0;

				if (normalize_depth) { // Additional gradients w.r.t. alpha from depth normalization
					dL_dchannel /= (1.f - T_final);
					dL_dalpha_depth += dL_dpixD * D_final / (1.f - T_final) / (1.f - T_final) * -T_final / (1 - alpha) / T;
				}
				
				dL_dalpha_depth += (d_cur - accum_rec_d) * dL_dchannel;

				// Update the gradients w.r.t. normal of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				// if (dL_dchannel != 0) printf("%.5f\n", dL_dchannel);
				atomicAdd(&(dL_ddepth[global_id]), dchannel_dcolor * dL_dchannel * 1);
				dL_dalpha += dL_dalpha_depth;
			}
			// }


			dL_dalpha *= T;
			
			// Add grads w.r.t. alpha from opac output
			dL_dalpha += dL_dpixO * T_final / (1 - alpha);
			
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixC[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;
			if (!normalize_depth) dL_dalpha += (-T_final / (1.f - alpha)) * (10 * dL_dpixD);
			
			float dL_ddist = 0, dL_dcut = 0;
			
			// if (use_cutoff && dist > cut - shift) {
			// 	dL_dalpha *= alpha_decay;
			// 	dL_dalpha_var *= T;
			// 	float dL_ddecay = (dL_dalpha_var) * (alpha / alpha_decay);
			// 	float dL_ddist_decay = dL_ddecay * alpha_decay * (alpha_decay - 1) * decay_weight;
			// 	float dL_dcut_decay = dL_ddecay * alpha_decay * (alpha_decay - 1) * -decay_weight;

			// 	// Add grad w.r.t. dist & cut from alpha decay
			// 	dL_ddist += dL_ddist_decay;
			// 	dL_dcut += dL_dcut_decay;

			// 	atomicAdd(&(dL_dcutoff[global_id]).x, dL_dcut);
			// 	// atomicAdd(&(dL_dcutoff[global_id]), dL_dcut);
			// }

			// printf("%.10f, ", dL_dalpha);

			dL_ddist += dL_dalpha * con_o.w * -0.5f * G;



			float2 dL_dNDC = {
				dL_ddist * 2 * (con_o.x * d.x + con_o.y * d.y) * ddelx_dx,
				dL_ddist * 2 * (con_o.z * d.y + con_o.y * d.x) * ddely_dy
			};
			float3 dL_dconic = {
				dL_ddist * (d.x * d.x),
				dL_ddist * (1 * d.x * d.y), // account for symmetry (x2) in computeCov2DCUDA(), here is not needed
				dL_ddist * (d.y * d.y)
			};

			// Add grads w.r.t. mean2D from differencing depth
			if (surface && per_pixel_depth) {
				dL_dNDC.x += 1 * -dL_dpixD * (Jinv_u0_u1[6] * Jinv_u0_u1[0] + Jinv_u0_u1[9] * Jinv_u0_u1[2]);
				dL_dNDC.y += 1 * -dL_dpixD * (Jinv_u0_u1[6] * Jinv_u0_u1[1] + Jinv_u0_u1[9] * Jinv_u0_u1[3]);
			}
			// printf("%.8f, %.8f\n", Jinv_u0_u1[2], Jinv_u0_u1[3]);

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dNDC.x);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dNDC.y);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, dL_dconic.x);
			atomicAdd(&dL_dconic2D[global_id].y, dL_dconic.y);
			atomicAdd(&dL_dconic2D[global_id].w, dL_dconic.z);

			// Update gradients w.r.t. opacity from mask loss
			float dL_dopac = G * dL_dalpha;
			atomicAdd(&(dL_dopacity[global_id]), dL_dopac);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dnormal,
	float* dL_ddepth,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dviewmat,
	float* dL_dprojmat,
	float* dL_dcampos,
	float* config)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D,
		dL_dviewmat,
		config);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dnormal,
		dL_ddepth,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot,
		dL_dviewmat,
		dL_dprojmat,
		dL_dcampos,
		config);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* patchbbox,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* normal,
	const float* depth,
	const float* Jinv,
	const float* viewCos,
	const float* final_Ts,
	const float* final_D,
	const float* final_C,
	const float* final_V,
	const uint32_t* n_contrib,
	const float* final_T_cut,
	const uint32_t* n_contrib_cut,
	const float* dL_dpixcolor,
	const float* dL_dpixnormal,
	const float* dL_dpixdepth,
	const float* dL_dpixopac,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_dnormal,
	float* dL_ddepth,
	float* config)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		patchbbox,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		normal,
		depth,
		Jinv,
		viewCos,
		final_Ts,
		final_D,
		final_C,
		final_V,
		n_contrib,
		final_T_cut,
		n_contrib_cut,
		dL_dpixcolor,
		dL_dpixnormal,
		dL_dpixdepth,
		dL_dpixopac,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_dnormal,
		dL_ddepth,
		config
		);
}