#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <unistd.h>
#include "cuda_utils.h"
using namespace torch::indexing;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define PI 3.141592654f
#define FOV_APPEND 0



std::vector<torch::Tensor> spatial_grouping(
    torch::Tensor worldPos,
    torch::Tensor normal, 
    torch::Tensor size,
    torch::Tensor opac,
    torch::Tensor group_feat,
    torch::Tensor group_count,
    torch::Tensor mean_feat,
    torch::Tensor cube_size,
    torch::Tensor cube_dim,
    torch::Tensor vtx_min,
    torch::Tensor R
    
) {

    CHECK_INPUT(worldPos);
    CHECK_INPUT(normal);
    CHECK_INPUT(size);
    CHECK_INPUT(opac);
    CHECK_INPUT(group_feat);
    CHECK_INPUT(group_count);
    CHECK_INPUT(mean_feat);
    CHECK_INPUT(cube_size);
    CHECK_INPUT(cube_dim);
    CHECK_INPUT(vtx_min);
    CHECK_INPUT(R);

    int n_point = worldPos.sizes()[0];

    group_feat.zero_();
    group_count.zero_();

    spatial_grouping_cuda(
        &n_point,
        worldPos.data_ptr<float>(), 
        normal.data_ptr<float>(), 
        size.data_ptr<float>(),  
        opac.data_ptr<float>(),  
        group_feat.data_ptr<float>(),
        group_count.data_ptr<int>(),
        mean_feat.data_ptr<float>(),
        cube_size.data_ptr<float>(), 
        cube_dim.data_ptr<int>(), 
        vtx_min.data_ptr<float>(), 
        R.data_ptr<float>()
    ); 


    auto out = std::vector<torch::Tensor>{};
    return out;
}

// 


torch::Tensor contour_padding(
    torch::Tensor image,
    torch::Tensor mask,
    int size
) {
    // torch::Device device(torch::kCUDA, 0);
    // auto camera = parse_camera(camera_in).to(device);
    
    int reso[3] = {image.sizes()[1], image.sizes()[2], image.sizes()[0]};
    torch::Tensor res = torch::zeros_like(image);
    if (size == 0) return res;
    CHECK_INPUT(image);
    CHECK_INPUT(mask);
    CHECK_INPUT(res);
    // // reso.index_put_({0}, reso0);
    // // reso.index_put_({1}, reso1);
    
    // // std::cout<<reso<<std::endl;

    // // float test = *torch::mean(d_img).to("cpu").data_ptr<float>();
    // // printf("d_img: %.3f\n", test);
    // // std::cout << d_img.sizes() << std::endl;

    contour_padding_cuda(
        image.data_ptr<float>(), 
        mask.data_ptr<bool>(),
        reso,
        res.data_ptr<float>(),
        &size
    );

    return res;
}


std::vector<torch::Tensor> point2tsdf(
    torch::Tensor worldPos,
    torch::Tensor normal, 
    torch::Tensor size,
    torch::Tensor config,
    torch::Tensor volgrid_d,
    torch::Tensor volgrid_w,
    torch::Tensor grid_range,
    torch::Tensor grid_dim_x,
    torch::Tensor truncate,
    torch::Tensor step_len,
    torch::Tensor tangent_size
) {

    CHECK_INPUT(worldPos);
    CHECK_INPUT(normal);
    CHECK_INPUT(size);
    CHECK_INPUT(config);
    CHECK_INPUT(volgrid_d);
    CHECK_INPUT(volgrid_w);
    CHECK_INPUT(grid_range);
    CHECK_INPUT(grid_dim_x);
    CHECK_INPUT(truncate);
    CHECK_INPUT(step_len);
    CHECK_INPUT(tangent_size);

    int n_point = worldPos.sizes()[0];

    point2tsdf_cuda(
        &n_point,
        worldPos.data_ptr<float>(),
        normal.data_ptr<float>(),
        size.data_ptr<float>(),
        config.data_ptr<float>(),
        volgrid_d.data_ptr<float>(),
        volgrid_w.data_ptr<float>(),
        grid_range.data_ptr<float>(),
        grid_dim_x.data_ptr<int>(),
        truncate.data_ptr<float>(),
        step_len.data_ptr<float>(),
        tangent_size.data_ptr<float>()
    );


    auto out = std::vector<torch::Tensor>{};
    return out;
}

std::vector<torch::Tensor> reprojection_flow(
    torch::Tensor worldPos0,
    torch::Tensor normal0, 
    // torch::Tensor depth0,
    // torch::Tensor config,
    torch::Tensor camera0,
    torch::Tensor camera1,
    // torch::Tensor normal1,
    torch::Tensor min_depth,
    torch::Tensor flow,
    torch::Tensor mask
) {

    CHECK_INPUT(worldPos0);
    CHECK_INPUT(normal0);
    // CHECK_INPUT(depth);
    // CHECK_INPUT(config);
    CHECK_INPUT(camera0);
    CHECK_INPUT(camera1);
    CHECK_INPUT(flow);
    CHECK_INPUT(min_depth);

    // int n_point = worldPos.sizes()[0];
    min_depth.zero_();
    mask.zero_();

    reprojection_flow_cuda(
        worldPos0.data_ptr<float>(),
        normal0.data_ptr<float>(),
        camera0.data_ptr<float>(),
        camera1.data_ptr<float>(),
        min_depth.data_ptr<float>(),
        flow.data_ptr<float>(),
        mask.data_ptr<bool>()
        
    );

    // min_depth = min_depth * mask;
    // flow = flow * mask;//.index({"...", None});

    auto out = std::vector<torch::Tensor>{};
    return out;
}

torch::Tensor gaussian2occgrid(
    torch::Tensor pos_min,
    torch::Tensor pos_max,
    torch::Tensor grid_len,
    torch::Tensor grid_dim,
    torch::Tensor pos,
    torch::Tensor rot,
    torch::Tensor scale,
    torch::Tensor opac,
    torch::Tensor cutoff
) {


    torch::Tensor grid = torch::zeros({grid_dim[0].item<int>(),
                                       grid_dim[1].item<int>(),
                                       grid_dim[2].item<int>()}).cuda().to(torch::kFloat32);
    int n_point = pos.sizes()[0];
    // gaussian2occgrid_cuda(
    //     &n_point,
    //     pos_min.data_ptr<float>(),
    //     pos_max.data_ptr<float>(),
    //     grid_len.data_ptr<float>(),
    //     grid_dim.data_ptr<int>(),
    //     grid.data_ptr<float>(),
    //     pos.data_ptr<float>(),
    //     rot.data_ptr<float>(),
    //     opac.data_ptr<float>()
    // );
    // CHECK_INPUT(pos_min);
    // CHECK_INPUT(pos_max);
    // CHECK_INPUT(grid_len);
    // CHECK_INPUT(grid_dim);
    // CHECK_INPUT(grid);
    // CHECK_INPUT(pos);
    // CHECK_INPUT(rot);
    // CHECK_INPUT(scale);
    // CHECK_INPUT(opac);

    gaussians2occgrid_cuda(
        &n_point, 
        pos_min.contiguous().data_ptr<float>(),
        pos_max.contiguous().data_ptr<float>(),
        grid_len.contiguous().data_ptr<float>(),
        grid_dim.contiguous().data_ptr<int>(),
        grid.contiguous().data_ptr<float>(),
        pos.contiguous().data_ptr<float>(),
        rot.contiguous().data_ptr<float>(),
        scale.contiguous().data_ptr<float>(),
        opac.contiguous().data_ptr<float>(),
        cutoff.contiguous().data_ptr<float>()
    );
    
    // grid = grid - 1;
    return grid;
}

torch::Tensor tsdf_fusion(
    torch::Tensor pos_min,
    torch::Tensor pos_max,
    torch::Tensor grid_len,
    torch::Tensor grid_dim,
    torch::Tensor pos,
    torch::Tensor normal,
    torch::Tensor center,
    torch::Tensor grid
) {
    int n_point = pos.sizes()[0];
    tsdf_fusion_cuda(
        &n_point, 
        pos_min.contiguous().data_ptr<float>(),
        pos_max.contiguous().data_ptr<float>(),
        grid_len.contiguous().data_ptr<float>(),
        grid_dim.contiguous().data_ptr<int>(),
        pos.contiguous().data_ptr<float>(),
        normal.contiguous().data_ptr<float>(),
        center.contiguous().data_ptr<float>(),
        grid.contiguous().data_ptr<float>()
    );

    return grid;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spatial_grouping", &spatial_grouping);
    m.def("contour_padding", &contour_padding);
    m.def("point2tsdf", &point2tsdf);
    m.def("reprojection_flow", &reprojection_flow);
    m.def("gaussian2occgrid", &gaussian2occgrid);
    m.def("tsdf_fusion", &tsdf_fusion);
}


