#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor query_ball_kernel_wrapper(torch::Tensor center_xyz, torch::Tensor xyz, torch::Tensor idx, float radius, int nsample);

torch::Tensor ball_query(torch::Tensor center_xyz, torch::Tensor xyz, const float radius, const int nsample) {
    CHECK_INPUT(center_xyz);
    CHECK_INPUT(xyz);

    torch::Tensor idx = torch::zeros({center_xyz.size(0), center_xyz.size(1), nsample}, center_xyz.options().dtype(torch::kInt64));

    return query_ball_kernel_wrapper(center_xyz, xyz, idx, radius, nsample);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ball_query", &ball_query, "ball_query");
}