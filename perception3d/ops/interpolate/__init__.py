import imp
import torch
import torch.nn as nn
from ..ball_query import ball_query
from torch.autograd import Function
import re
from glob import glob
import os
from perception3d.utils import build_ext
from ..ball_query import square_distance
from ..group_points import index_points_nocuda
import warnings

has_ext = True
if torch.cuda.is_available():
    try:
        from . import interpolate_ext
    except (ImportError, ModuleNotFoundError):
        build_ext('interpolate_ext', os.path.dirname(__file__))
        try:
            from . import interpolate_ext
        except (ImportError, ModuleNotFoundError):
            warnings.warn('Error building extension for {}'.format(os.path.basename(os.path.dirname(__file__))))
            has_ext = False


import torch
from torch.autograd import Function
from typing import Tuple


class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, indices: torch.Tensor,
                weight: torch.Tensor) -> torch.Tensor:
        """Performs weighted linear interpolation on 3 features.
        Args:
            features (Tensor): (B, C, M) Features descriptors to be
                interpolated from
            indices (Tensor): (B, n, 3) index three nearest neighbors
                of the target features in features
            weight (Tensor): (B, n, 3) weights of interpolation
        Returns:
            Tensor: (B, C, N) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert indices.is_contiguous()
        assert weight.is_contiguous()

        ctx.three_interpolate_for_backward = (
            indices, weight, features.shape[2])
        output = interpolate_ext.three_interpolate_forward(
            features, indices, weight)
        return output

    @staticmethod
    def backward(
        ctx, grad_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward of three interpolate.
        Args:
            grad_out (Tensor): (B, C, N) tensor with gradients of outputs
        Returns:
            Tensor: (B, C, M) tensor with gradients of features
        """
        idx, weight, m = ctx.three_interpolate_for_backward

        grad_features = interpolate_ext.three_interpolate_backward(
            grad_out.contiguous(), idx, weight, m)
        return grad_features, None, None


def three_interpolate_nocuda(features, indices, weight):
    interpolated_points = torch.sum(index_points_nocuda(features, indices) * weight[:, None], dim=-1)
    return interpolated_points


three_interpolate = ThreeInterpolate.apply if has_ext else three_interpolate_nocuda


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, target: torch.Tensor,
                source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find the top-3 nearest neighbors of the target set from the source
        set.
        Args:
            target (Tensor): shape (B, N, 3), points set that needs to
                find the nearest neighbors.
            source (Tensor): shape (B, M, 3), points set that is used
                to find the nearest neighbors of points in target set.
        Returns:
            Tensor: shape (B, N, 3), L2 distance of each point in target
                set to their corresponding nearest neighbors.
        """
        assert target.is_contiguous()
        assert source.is_contiguous()

        dist, idx = interpolate_ext.three_nn_forward(target, source)

        ctx.mark_non_differentiable(idx)

        return dist, idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


def three_nn_nocuda(src, dst):
    dists = torch.sqrt(square_distance(src, dst))
    dists, idx = dists.sort(dim=-1)
    dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
    return dists, idx


three_nn = ThreeNN.apply if has_ext else three_nn_nocuda


__all__ = ['three_nn', 'three_interpolate']