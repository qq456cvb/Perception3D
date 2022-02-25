#include <torch/extension.h>
#include "../../_common/macro.h"
#include <iostream>

torch::Tensor ball_query_cuda_forward(torch::Tensor xyz, torch::Tensor center_xyz, torch::Tensor idx, float radius, int nsample);

torch::Tensor ball_query_forward(torch::Tensor xyz, torch::Tensor center_xyz, const float radius, const int nsample)
{
    CHECK_INPUT(center_xyz);
    CHECK_INPUT(xyz);

    torch::Tensor idx = torch::zeros({center_xyz.size(0), center_xyz.size(1), nsample}, center_xyz.options().dtype(torch::kInt64));
    return ball_query_cuda_forward(xyz, center_xyz, idx, radius, nsample);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ball_query_forward", &ball_query_forward, "ball_query_forward");
}