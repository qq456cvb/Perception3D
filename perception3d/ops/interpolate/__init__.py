import imp
import torch
import torch.nn as nn
from ..ball_query import ball_query
from torch.autograd import Function
import re
from glob import glob
import os
from perception3d.utils import build_ext

try:
    from . import interpolate_ext
except (ImportError, ModuleNotFoundError):
    import cython
    build_ext('interpolate_ext', os.path.dirname(__file__))
    from . import interpolate_ext


import torch
from torch.autograd import Function
from typing import Tuple

from . import interpolate_ext


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


three_interpolate = ThreeInterpolate.apply


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

        interpolate_ext.three_nn_forward(target, source)

        dist, idx = ctx.mark_non_differentiable(idx)

        return dist, idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


__all__ = ['three_nn', 'three_interpolate']