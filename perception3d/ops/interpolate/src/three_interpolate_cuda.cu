#include <torch/extension.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../_common/macro.h"

template <typename scalar_t>
__global__ void three_interpolate_cuda_forward_kernel(ACCESSOR(scalar_t, 3) features, ACCESSOR(int64_t, 3) idx, ACCESSOR(scalar_t, 3) weight, ACCESSOR(scalar_t, 3) out)
{
    // points: (B, C, M)
    // idx: (B, N, 3)
    // weight: (B, N, 3)
    // output:
    //      out: (B, C, N)

    int b_idx = blockIdx.y;
    int c_idx = blockIdx.z;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b_idx >= features.size(0) || c_idx >= features.size(1) || pt_idx >= idx.size(1))
        return;

    out[b_idx][c_idx][pt_idx] = weight[b_idx][pt_idx][0] * features[b_idx][c_idx][idx[b_idx][pt_idx][0]] + weight[b_idx][pt_idx][1] * features[b_idx][c_idx][idx[b_idx][pt_idx][1]] + weight[b_idx][pt_idx][2] * features[b_idx][c_idx][idx[b_idx][pt_idx][2]];
}

void three_interpolate_cuda_forward(torch::Tensor features, torch::Tensor idx, torch::Tensor weight, torch::Tensor out)
{
    // points: (B, C, M)
    // idx: (B, N, 3)
    // weight: (B, N, 3)
    // output:
    //      out: (B, C, N)
    dim3 blocks((idx.size(0) - 1) / THREADS_PER_BLOCK + 1, features.size(0),
                features.size(1)); // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES(features.scalar_type(), "three_interpolate_cuda_forward_kernel", ([&]
                                                                                                 { three_interpolate_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                                                                                                       features.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                                                       idx.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
                                                                                                       weight.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                                                       out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()); }));
}

template <typename scalar_t>
__global__ void three_interpolate_cuda_backward_kernel(
    ACCESSOR(scalar_t, 3) grad_out, ACCESSOR(int64_t, 3) idx, ACCESSOR(scalar_t, 3) weight, ACCESSOR(scalar_t, 3) grad_features)
{
    // grad_out: (B, C, N)
    // weight: (B, N, 3)
    // output:
    //      grad_points: (B, C, M)

    int b_idx = blockIdx.y;
    int c_idx = blockIdx.z;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b_idx >= grad_out.size(0) || c_idx >= grad_out.size(1) || pt_idx >= grad_out.size(2))
        return;

    atomicAdd(&grad_features[b_idx][c_idx][idx[b_idx][pt_idx][0]], grad_out[b_idx][c_idx][pt_idx] * weight[b_idx][pt_idx][0]);
    atomicAdd(&grad_features[b_idx][c_idx][idx[b_idx][pt_idx][1]], grad_out[b_idx][c_idx][pt_idx] * weight[b_idx][pt_idx][1]);
    atomicAdd(&grad_features[b_idx][c_idx][idx[b_idx][pt_idx][2]], grad_out[b_idx][c_idx][pt_idx] * weight[b_idx][pt_idx][2]);
}

void three_interpolate_cuda_backward(torch::Tensor grad_out, torch::Tensor idx, torch::Tensor weight, torch::Tensor grad_features)
{
    // grad_out: (B, C, N)
    // weight: (B, N, 3)
    // output:
    //      grad_points: (B, C, M)

    dim3 blocks((grad_out.size(2) - 1) / THREADS_PER_BLOCK + 1, grad_out.size(0),
                grad_out.size(1)); // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "three_interpolate_cuda_backward_kernel", ([&]
                                                                                                  { three_interpolate_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                                                                                                        grad_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                                                        idx.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
                                                                                                        weight.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                                                        grad_features.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()); }));
}