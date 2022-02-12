from glob import glob
import re
from perceptron3d.utils import build_ext
import os

try:
    from . import ball_query_ext
except (ImportError, ModuleNotFoundError):
    import cython
    build_ext('ball_query_ext', os.path.dirname(__file__))
    from . import ball_query_ext

import torch
from torch.autograd import Function


class BallQuery(Function):
    @staticmethod
    def forward(ctx, center_xyz, xyz, radius, num_sample):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query
        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        output = ball_query_ext.ball_query(center_xyz, xyz, radius, num_sample)

        ctx.mark_non_differentiable(output)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        return ()


ball_query = BallQuery.apply

__all__ = ['ball_query']