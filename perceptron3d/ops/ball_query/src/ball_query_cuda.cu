
#include <torch/extension.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
template <typename scalar_t>
__global__ void query_ball_point_kernel(torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> center_xyz,
                                        torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> xyz,
                                        torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> idx,
                                        float radius, int nsample)
{
    int bidx = blockIdx.x;
    int index = threadIdx.x;
    int stride = blockDim.x;

    int m = center_xyz.size(1);
    int n = xyz.size(1);
    float radius2 = radius * radius;
    for (int j = index; j < m; j += stride)
    {
        float center_x = center_xyz[bidx][j][0];
        float center_y = center_xyz[bidx][j][1];
        float center_z = center_xyz[bidx][j][2];
        for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k)
        {
            float x = xyz[bidx][k][0];
            float y = xyz[bidx][k][1];
            float z = xyz[bidx][k][2];
            float d2 = (center_x - x) * (center_x - x) + (center_y - y) * (center_y - y) +
                       (center_z - z) * (center_z - z);
            if (d2 < radius2)
            {
                if (cnt == 0)
                {
                    for (int l = 0; l < nsample; ++l)
                    {
                        idx[bidx][j][l] = k;
                    }
                }
                idx[bidx][j][cnt] = k;
                ++cnt;
            }
        }
    }
}

torch::Tensor query_ball_kernel_wrapper(torch::Tensor center_xyz, torch::Tensor xyz, torch::Tensor idx, float radius, int nsample)
{
    int threads_per_block = 256;
    dim3 blocks(center_xyz.size(0));
    dim3 threads((center_xyz.size(1) - 1) / threads_per_block + 1);
    AT_DISPATCH_FLOATING_TYPES(xyz.scalar_type(), "hv_backward_cuda", ([&]
                                                                       { query_ball_point_kernel<scalar_t><<<blocks, threads>>>(
                                                                             center_xyz.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                             xyz.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                             idx.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
                                                                             radius, nsample); }));
    return idx;
}