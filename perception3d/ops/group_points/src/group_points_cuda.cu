#include <torch/extension.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../_common/macro.h"

// input: points(b, c, n) idx(b, npoints, nsample)
// output: out(b, c, npoints, nsample)
template <typename scalar_t>
__global__ void group_points_cuda_forward_kernel(ACCESSOR(scalar_t, 3) points,
                                                 ACCESSOR(int64_t, 3) idx, ACCESSOR(scalar_t, 4) out)
{
    int b_idx = blockIdx.y;
    int c_idx = blockIdx.z;
    int pt_sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_sample = idx.size(2);
    int pt_idx = pt_sample_idx / n_sample;
    if (b_idx >= points.size(0) || c_idx >= points.size(1) || pt_idx >= idx.size(1))
        return;
    int sample_idx = pt_sample_idx % n_sample;

    out[b_idx][c_idx][pt_idx][sample_idx] = points[b_idx][c_idx][idx[b_idx][pt_idx][sample_idx]];
}

void group_points_cuda_forward(torch::Tensor points, torch::Tensor idx, torch::Tensor out)
{
    dim3 blocks((idx.size(1) * idx.size(2) - 1) / THREADS_PER_BLOCK + 1, points.size(0), points.size(1));
    dim3 threads(THREADS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "group_points_cuda_forward_kernel", ([&]
                                                                                          { group_points_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                                                                                                points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                                                idx.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
                                                                                                out.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()); }));
}

// input: grad_out(b, c, npoints, nsample), idx(b, npoints, nsample)
// output: grad_points(b, c, n)
template <typename scalar_t>
__global__ void group_points_cuda_backward_kernel(ACCESSOR(scalar_t, 3) grad_points, ACCESSOR(int64_t, 3) idx, ACCESSOR(scalar_t, 4) grad_out)
{
    int b_idx = blockIdx.y;
    int c_idx = blockIdx.z;
    int pt_sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_sample = idx.size(2);
    int pt_idx = pt_sample_idx / n_sample;
    if (b_idx >= grad_points.size(0) || c_idx >= grad_points.size(1) || pt_idx >= idx.size(1))
        return;
    int sample_idx = pt_sample_idx % n_sample;

    atomicAdd(&grad_points[b_idx][c_idx][idx[b_idx][pt_idx][sample_idx]], grad_out[b_idx][c_idx][pt_idx][sample_idx]);
}

void group_points_cuda_backward(torch::Tensor grad_points, torch::Tensor idx, torch::Tensor grad_out)
{
    dim3 blocks(((idx.size(1) * idx.size(2)) - 1) / THREADS_PER_BLOCK + 1, grad_points.size(0), grad_points.size(1));
    dim3 threads(THREADS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES(grad_points.scalar_type(), "group_points_cuda_backward_kernel", ([&]
                                                                                                { group_points_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                                                                                                      grad_points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                                                      idx.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
                                                                                                      grad_out.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()); }));
}