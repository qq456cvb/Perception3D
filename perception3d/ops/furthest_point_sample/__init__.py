import imp
import torch
import torch.nn as nn
from torch.autograd import Function
import re
from glob import glob
import os
from perception3d.utils import build_ext

try:
    from . import furthest_point_sample_ext
except (ImportError, ModuleNotFoundError):
    import cython
    build_ext('furthest_point_sample_ext', os.path.dirname(__file__))
    from . import furthest_point_sample_ext


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

        output = furthest_point_sample_ext.furthest_point_sampling_forward(points_xyz)
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


furthest_point_sample = FurthestPointSampling.apply
furthest_point_sample_with_dist = FurthestPointSamplingWithDist.apply

__all__ = ['furthest_point_sample', 'furthest_point_sample_with_dist']