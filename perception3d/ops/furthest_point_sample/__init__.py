import imp
import torch
import torch.nn as nn
from torch.autograd import Function
import re
from glob import glob
import os
from perception3d.utils import build_ext
import warnings


has_ext = True
if torch.cuda.is_available():
    try:
        from . import furthest_point_sample_ext
    except (ImportError, ModuleNotFoundError):
        build_ext('furthest_point_sample_ext', os.path.dirname(__file__))
        try:
            from . import furthest_point_sample_ext
        except (ImportError, ModuleNotFoundError):
            warnings.warn('Error building extension for {}'.format(os.path.basename(os.path.dirname(__file__))))
            has_ext = False


class FurthestPointSampling(Function):
    """Furthest Point Sampling.
    Uses iterative furthest point sampling to select a set of features whose
    corresponding points have the furthest distance.
    """

    @staticmethod
    def forward(ctx, points_xyz: torch.Tensor,
                num_points: int) -> torch.Tensor:
        """forward.
        Args:
            points_xyz (Tensor): (B, N, 3) where N > num_points.
            num_points (int): Number of points in the sampled set.
        Returns:
             Tensor: (B, num_points) indices of the sampled points.
        """
        assert points_xyz.is_contiguous()

        output = furthest_point_sample_ext.furthest_point_sampling_forward(points_xyz, num_points)
        ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


class FurthestPointSamplingWithDist(Function):
    """Furthest Point Sampling With Distance.
    Uses iterative furthest point sampling to select a set of features whose
    corresponding points have the furthest distance.
    """

    @staticmethod
    def forward(ctx, points_dist: torch.Tensor,
                num_points: int) -> torch.Tensor:
        """forward.
        Args:
            points_dist (Tensor): (B, N, N) Distance between each point pair.
            num_points (int): Number of points in the sampled set.
        Returns:
             Tensor: (B, num_points) indices of the sampled points.
        """
        assert points_dist.is_contiguous()

        output = furthest_point_sample_ext.furthest_point_sampling_with_dist_forward(points_dist, num_points)
        ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
def furthest_point_sample_nocuda(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


furthest_point_sample = FurthestPointSampling.apply if has_ext else furthest_point_sample_nocuda
furthest_point_sample_with_dist = FurthestPointSamplingWithDist.apply

__all__ = ['furthest_point_sample', 'furthest_point_sample_with_dist']