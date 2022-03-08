// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/sampling.cpp

#include <torch/extension.h>
#include "../../_common/macro.h"
#include <vector>

void furthest_point_sampling_cuda_forward(torch::Tensor points,
                                         torch::Tensor temp,
                                         torch::Tensor idx);

void furthest_point_sampling_with_dist_cuda_forward(torch::Tensor points_dist,
                                                   torch::Tensor temp,
                                                   torch::Tensor idx);

torch::Tensor furthest_point_sampling_forward(torch::Tensor points, int num_points)
{
    CHECK_INPUT(points);

    // TODO: hardcode 1e10 here, possibly determine max by type?
    torch::Tensor temp = torch::full({points.size(0), points.size(1)}, 1e10, points.options());
    torch::Tensor out = torch::zeros({points.size(0), num_points}, points.options().dtype(torch::kInt64));

    furthest_point_sampling_cuda_forward(points, temp, out);
    return out;
}

torch::Tensor furthest_point_sampling_with_dist_forward(torch::Tensor points_dist, int num_points)
{
    CHECK_INPUT(points_dist);

    torch::Tensor temp = torch::full({points_dist.size(0), points_dist.size(1)}, 1e10, points_dist.options());
    torch::Tensor out = torch::zeros({points_dist.size(0), num_points}, points_dist.options().dtype(torch::kInt64));

    furthest_point_sampling_with_dist_cuda_forward(points_dist, temp, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("furthest_point_sampling_forward", &furthest_point_sampling_forward,
          "furthest_point_sampling_forward");
    m.def("furthest_point_sampling_with_dist_forward",
          &furthest_point_sampling_with_dist_forward,
          "furthest_point_sampling_with_dist_forward");
}