#include <torch/extension.h>

#define THREADS_PER_BLOCK 256
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define ACCESSOR(type, dim) torch::PackedTensorAccessor32<type, dim, torch::RestrictPtrTraits>