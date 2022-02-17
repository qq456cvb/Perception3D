#include <torch/extension.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../_common/macro.h"

template <typename scalar_t>
__global__ void three_nn_cuda_forward_kernel(ACCESSOR(scalar_t, 3) unknown, ACCESSOR(scalar_t, 3) known, ACCESSOR(scalar_t, 3) dist, ACCESSOR(int64_t, 3) idx)
{
    // unknown: (B, N, 3)
    // known: (B, M, 3)
    // output:
    //      dist2: (B, N, 3)
    //      idx: (B, N, 3)

    int b_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (b_idx >= unknown.size(0) || pt_idx >= unknown.size(1))
        return;

    float ux = unknown[b_idx][pt_idx][0];
    float uy = unknown[b_idx][pt_idx][1];
    float uz = unknown[b_idx][pt_idx][2];

    float best1 = FLT_MAX, best2 = FLT_MAX, best3 = FLT_MAX;
    int besti1 = 0, besti2 = 0, besti3 = 0;
    for (int k = 0; k < known.size(1); ++k)
    {
        float x = known[b_idx][k][0];
        float y = known[b_idx][k][1];
        float z = known[b_idx][k][2];
        float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
        if (d < best1)
        {
            best3 = best2;
            besti3 = besti2;
            best2 = best1;
            besti2 = besti1;
            best1 = d;
            besti1 = k;
        }
        else if (d < best2)
        {
            best3 = best2;
            besti3 = besti2;
            best2 = d;
            besti2 = k;
        }
        else if (d < best3)
        {
            best3 = d;
            besti3 = k;
        }
    }
    dist[b_idx][pt_idx][0] = sqrt(best1);
    dist[b_idx][pt_idx][1] = sqrt(best2);
    dist[b_idx][pt_idx][2] = sqrt(best3);
    idx[b_idx][pt_idx][0] = besti1;
    idx[b_idx][pt_idx][1] = besti2;
    idx[b_idx][pt_idx][2] = besti3;
}

void three_nn_cuda_forward(torch::Tensor unknown, torch::Tensor known, torch::Tensor dist, torch::Tensor idx)
{
    // unknown: (B, N, 3)
    // known: (B, M, 3)
    // output:
    //      dist2: (B, N, 3)
    //      idx: (B, N, 3)

    dim3 blocks((unknown.size(1) - 1) / THREADS_PER_BLOCK + 1, unknown.size(0));
    dim3 threads(THREADS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES(unknown.scalar_type(), "three_nn_cuda_forward_kernel", ([&]
                                                                                       { three_nn_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                                                                                             unknown.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                                             known.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                                             dist.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                                             idx.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>()); }));
}
