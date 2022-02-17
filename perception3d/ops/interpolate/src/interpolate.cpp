#include <torch/extension.h>
#include "../../_common/macro.h"
#include <vector>

void three_nn_cuda_forward(torch::Tensor unknown, torch::Tensor known, torch::Tensor dist, torch::Tensor idx);

std::vector<torch::Tensor> three_nn_forward(torch::Tensor unknown, torch::Tensor known)
{
    CHECK_INPUT(unknown);
    CHECK_INPUT(known);

    torch::Tensor dist = torch::zeros({unknown.size(0), unknown.size(1), 3}, unknown.options());
    torch::Tensor idx = torch::zeros({unknown.size(0), unknown.size(1), 3}, unknown.options().dtype(torch::kInt64));

    three_nn_cuda_forward(unknown, known, dist, idx);
    return {dist, idx};
}

void three_interpolate_cuda_forward(torch::Tensor features, torch::Tensor idx, torch::Tensor weight, torch::Tensor out);

torch::Tensor three_interpolate_forward(torch::Tensor features, torch::Tensor idx, torch::Tensor weight)
{
    CHECK_INPUT(features);
    CHECK_INPUT(idx);
    CHECK_INPUT(weight);

    assert(idx.scalar_type() == torch::kInt64);

    torch::Tensor out = torch::zeros({features.size(0), features.size(1), idx.size(1)}, features.options());

    three_interpolate_cuda_forward(features, idx, weight, out);
    return out;
}

void three_interpolate_cuda_backward(torch::Tensor grad_out, torch::Tensor idx, torch::Tensor weight, torch::Tensor grad_features);

torch::Tensor three_interpolate_backward(torch::Tensor grad_out, torch::Tensor idx, torch::Tensor weight, int m)
{
    CHECK_INPUT(grad_out);
    CHECK_INPUT(idx);
    CHECK_INPUT(weight);

    assert(idx.scalar_type() == torch::kInt64);

    torch::Tensor grad_features = torch::zeros({grad_out.size(0), grad_out.size(1), m}, grad_out.options());

    three_interpolate_cuda_backward(grad_out, idx, weight, grad_features);
    return grad_features;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("three_nn_forward", &three_nn_forward, "three_nn_forward");
    m.def("three_interpolate_forward", &three_interpolate_forward,
          "three_interpolate_forward");
    m.def("three_interpolate_backward", &three_interpolate_backward,
          "three_interpolate_backward");
}