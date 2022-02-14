#include <torch/extension.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../_common/macro.h"

#define TOTAL_THREADS 1024
#define KERNEL_CASE(n, kernel, scalar_t, blocks, threads, ...)\
    case n: {\
        kernel<n, scalar_t><<<blocks, threads>>>(__VA_ARGS__);\
    }
#define DISPATCH_KERNEL(n_threads, kernel, scalar_t, blocks, threads, ...)\
    switch (n_threads) {\
        KERNEL_CASE(1024, kernel, scalar_t, blocks, threads, __VA_ARGS__)\
        KERNEL_CASE(512, kernel, scalar_t, blocks, threads, __VA_ARGS__)\
        KERNEL_CASE(256, kernel, scalar_t, blocks, threads, __VA_ARGS__)\
        KERNEL_CASE(128, kernel,scalar_t, blocks, threads, __VA_ARGS__)\
        KERNEL_CASE(64, kernel, scalar_t, blocks, threads, __VA_ARGS__)\
        KERNEL_CASE(32, kernel, scalar_t, blocks, threads, __VA_ARGS__)\
        KERNEL_CASE(16, kernel, scalar_t, blocks, threads, __VA_ARGS__)\
        KERNEL_CASE(8, kernel, scalar_t, blocks, threads, __VA_ARGS__)\
        KERNEL_CASE(4, kernel, scalar_t, blocks, threads, __VA_ARGS__)\
        KERNEL_CASE(2, kernel, scalar_t, blocks, threads, __VA_ARGS__)\
        KERNEL_CASE(1, kernel, scalar_t, blocks, threads, __VA_ARGS__)\
        default: ;\
    }

inline int opt_n_threads(int work_size)
{
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

    return max(min(1 << pow_2, TOTAL_THREADS), 1);
}

__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
                         int idx1, int idx2)
{
    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}

#define SYNC_UPDATE(n)\
    {\
        int m = n / 2;\
        if (block_size >= n)\
        {\
            if (tid < m)\
            {\
                __update(dists, dists_i, tid, tid + m);\
            }\
            __syncthreads();\
        }\
    }

template <unsigned int block_size, typename scalar_t>
__global__ void furthest_point_sampling_cuda_forward_kernel(ACCESSOR(scalar_t, 3) points, ACCESSOR(scalar_t, 2) temp, ACCESSOR(int64_t, 2) idx)
{
    // dataset: (B, N, 3)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    // if (m <= 0)
    //     return;
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    int b_idx = blockIdx.x;
    int tid = threadIdx.x;
    const int stride = block_size;

    int old = 0;
    if (threadIdx.x == 0)
        idx[b_idx][0] = old;

    __syncthreads();
    for (int j = 1; j < idx.size(1); j++)
    {
        int besti = 0;
        float best = -1;
        float x1 = points[b_idx][old][0];
        float y1 = points[b_idx][old][1];
        float z1 = points[b_idx][old][2];
        for (int k = tid; k < points.size(1); k += stride)
        {
            float x2, y2, z2;
            x2 = points[b_idx][k][0];
            y2 = points[b_idx][k][0];
            z2 = points[b_idx][k][0];
            // float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
            // if (mag <= 1e-3)
            // continue;

            float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
            float d2 = min(d, temp[b_idx][k]);
            temp[b_idx][k] = d2;
            besti = d2 > best ? k : besti;
            best = d2 > best ? d2 : best;
        }
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();

        SYNC_UPDATE(1024);
        SYNC_UPDATE(512);
        SYNC_UPDATE(256);
        SYNC_UPDATE(128);
        SYNC_UPDATE(64);
        SYNC_UPDATE(32);
        SYNC_UPDATE(16);
        SYNC_UPDATE(8);
        SYNC_UPDATE(4);
        SYNC_UPDATE(2);

        old = dists_i[0];
        if (tid == 0)
            idx[b_idx][j] = old;
    }
}

void furthest_point_sampling_cuda_forward(torch::Tensor points,
    torch::Tensor temp,
    torch::Tensor idx)
{
    // dataset: (B, N, 3)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    unsigned int n_threads = opt_n_threads(points.size(1));

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "furthest_point_sampling_cuda_forward_kernel", ([&] 
        {
            DISPATCH_KERNEL(n_threads, furthest_point_sampling_cuda_forward_kernel, scalar_t, points.size(0), n_threads, 
            points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            temp.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            idx.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>()
            );
        }));
}

// Modified from
// https://github.com/qiqihaer/3DSSD-pytorch/blob/master/lib/pointnet2/src/sampling_gpu.cu
template <unsigned int block_size, typename scalar_t>
__global__ void furthest_point_sampling_with_dist_cuda_forward_kernel(ACCESSOR(scalar_t, 3) points_dist, ACCESSOR(scalar_t, 2) temp, ACCESSOR(int64_t, 2) idx)
{
    // dataset: (B, N, N)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    int b_idx = blockIdx.x;

    int tid = threadIdx.x;
    const int stride = block_size;

    int old = 0;
    if (threadIdx.x == 0)
        idx[b_idx][0] = old;

    __syncthreads();
    for (int j = 1; j < idx.size(1); j++)
    {
        int besti = 0;
        float best = -1;
        for (int k = tid; k < points_dist.size(2); k += stride)
        {
            float d = points_dist[b_idx][old][k];

            float d2 = min(d, temp[b_idx][k]);
            temp[b_idx][k] = d2;
            besti = d2 > best ? k : besti;
            best = d2 > best ? d2 : best;
        }
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();

        SYNC_UPDATE(1024);
        SYNC_UPDATE(512);
        SYNC_UPDATE(256);
        SYNC_UPDATE(128);
        SYNC_UPDATE(64);
        SYNC_UPDATE(32);
        SYNC_UPDATE(16);
        SYNC_UPDATE(8);
        SYNC_UPDATE(4);
        SYNC_UPDATE(2);

        old = dists_i[0];
        if (tid == 0)
            idx[b_idx][j] = old;
    }
}

void furthest_point_sampling_with_dist_cuda_forward(torch::Tensor points_dist,
    torch::Tensor temp,
    torch::Tensor idx)
{
    // dataset: (B, N, N)
    // temp: (B, N)
    // output:
    //      idx: (B, M)

    unsigned int n_threads = opt_n_threads(points_dist.size(2));

    AT_DISPATCH_FLOATING_TYPES(points_dist.scalar_type(), "furthest_point_sampling_with_dist_cuda_forward_kernel", ([&] 
        {
            DISPATCH_KERNEL(n_threads, furthest_point_sampling_with_dist_cuda_forward_kernel, scalar_t, points_dist.size(0), n_threads, 
            points_dist.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            temp.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            idx.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>()
            );
        }));
}