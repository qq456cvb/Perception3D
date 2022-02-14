#include <torch/extension.h>
#include "../../_common/macro.h"

void group_points_cuda_forward(torch::Tensor points, torch::Tensor idx, torch::Tensor out);

void group_points_cuda_backward(torch::Tensor grad_points, torch::Tensor idx, torch::Tensor grad_out);

torch::Tensor group_points_forward(torch::Tensor points, torch::Tensor idx)
{
    CHECK_INPUT(points);
    CHECK_INPUT(idx);
    assert(idx.scalar_type() == torch::kInt64);

    torch::Tensor output = torch::zeros({points.size(0), points.size(1), idx.size(1), idx.size(2)}, points.options());

    group_points_cuda_forward(points, idx, output);

    return output;
}

torch::Tensor group_points_backward(torch::Tensor grad_out, torch::Tensor idx, int n)
{
    CHECK_INPUT(grad_out);
    CHECK_INPUT(idx);

    torch::Tensor output = torch::zeros({grad_out.size(0), grad_out.size(1), n}, grad_out.options());

    group_points_cuda_backward(grad_out, idx, output);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("group_points_forward", &group_points_forward, "group_points_forward");
    m.def("group_points_backward", &group_points_backward, "group_points_backward");
}