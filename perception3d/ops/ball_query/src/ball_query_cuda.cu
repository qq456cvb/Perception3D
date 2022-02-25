
#include <torch/extension.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../_common/macro.h"

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
template <typename scalar_t>
__global__ void ball_query_cuda_forward_kernel(ACCESSOR(scalar_t, 3) xyz,
                                               ACCESSOR(scalar_t, 3) center_xyz,
                                               ACCESSOR(int64_t, 3) idx,
                                               float radius, int nsample)
{
    int b_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int b = center_xyz.size(0);
    int m = center_xyz.size(1);
    if (b_idx >= b || pt_idx >= m)
        return;

    int n = xyz.size(1);
    float radius2 = radius * radius;
    float center_x = center_xyz[b_idx][pt_idx][0];
    float center_y = center_xyz[b_idx][pt_idx][1];
    float center_z = center_xyz[b_idx][pt_idx][2];
    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k)
    {
        float x = xyz[b_idx][k][0];
        float y = xyz[b_idx][k][1];
        float z = xyz[b_idx][k][2];
        float d2 = (center_x - x) * (center_x - x) + (center_y - y) * (center_y - y) +
                   (center_z - z) * (center_z - z);
        if (d2 < radius2)
        {
            if (cnt == 0)
            {
                for (int l = 0; l < nsample; ++l)
                {
                    idx[b_idx][pt_idx][l] = k;
                }
            }
            idx[b_idx][pt_idx][cnt] = k;
            ++cnt;
        }
    }
}

torch::Tensor ball_query_cuda_forward(torch::Tensor xyz, torch::Tensor center_xyz, torch::Tensor idx, float radius, int nsample)
{
    dim3 blocks((center_xyz.size(1) - 1) / THREADS_PER_BLOCK + 1, center_xyz.size(0));
    dim3 threads(THREADS_PER_BLOCK);
    AT_DISPATCH_FLOATING_TYPES(xyz.scalar_type(), "ball_query_cuda_forward_kernel", ([&]
                                                                                     { ball_query_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                                                                                           xyz.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                                           center_xyz.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                                           idx.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
                                                                                           radius, nsample); }));
    CHECK_ERROR();
    return idx;
}