#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>

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
);

void contour_padding_cuda(
    float* image,
    bool* mask,
    int* reso,
    float* res,
    int* size
);

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
);

void reprojection_flow_cuda(
    float* worldPos0,
    float* normal0,
    float* camera0,
    float* camera1,
    float* min_depth,
    float* flow,
    bool* mask
);


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
);

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
);