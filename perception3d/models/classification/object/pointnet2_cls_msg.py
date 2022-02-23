from typing import List
import torch
from torch import nn as nn
from perception3d.models._modules.N2n.point_sa_module import PointSAModuleMSG
from perception3d.models._modules.n2N.point_fp_module import PointFPModule


class PointNet2ClsMSG(nn.Module):
    def __init__(self,
                 in_channels,
                 num_class,
                 sa_modules: List[PointSAModuleMSG]=[],
                 out_mlp_channels=[],
                 out_dropouts=[]
                 ):
        super().__init__()
        self.num_sa = len(sa_modules)
        self.num_class = num_class

        self.SA_modules = nn.ModuleList()
        self.out_mlps = nn.ModuleList()
        sa_in_channel = in_channels - 3  # number of channels without xyz

        for sa_index in range(self.num_sa):
            cur_sa_mlps = list(sa_modules[sa_index].mlp_channels)
            sa_out_channel = 0
            for radius_index in range(len(sa_modules[sa_index].radii[sa_index])):
                cur_sa_mlps[radius_index] = [sa_in_channel] + list(
                    cur_sa_mlps[radius_index])
                sa_out_channel += cur_sa_mlps[radius_index][-1]
                sa_modules[sa_index].mlp_channels = cur_sa_mlps

            self.SA_modules.append(sa_modules[sa_index])
            sa_in_channel = sa_out_channel
            
        if sa_out_channel != out_mlp_channels[0]:
            out_mlp_channels = [sa_out_channel] + out_mlp_channels
            out_dropouts = [0.] + out_dropouts
        for i in range(len(out_mlp_channels) - 1):
            self.out_mlps.append(nn.Linear(out_mlp_channels[i], out_mlp_channels[i + 1]))
            self.out_mlps.append(nn.BatchNorm1d(out_mlp_channels[i + 1]))
            self.out_mlps.append(nn.Dropout(out_dropouts[i]))
        
        self.out_mlps.append(nn.Linear(out_mlp_channels[-1], num_class))

    def forward(self, inputs):
        """Forward pass.
        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).
        Returns:
            dict[str, torch.Tensor]: Outputs of the last SA module.
                - sa_xyz (torch.Tensor): The coordinates of sa features.
                - sa_features (torch.Tensor): The features from the
                    last Set Aggregation Layers.
                - sa_indices (torch.Tensor): Indices of the \
                    input points.
        """
        cur_xyz = inputs['points']
        if 'features' in inputs:
            cur_features = inputs['features']
        else:
            cur_features = None

        for i in range(self.num_sa):
            cur_xyz, cur_features, cur_indices = self.SA_modules[i](cur_xyz, cur_features)
        
        features = cur_features.reshape(cur_features.shape[0], -1)
        label_logit = self.out_mlps(features)
        return {'label_logit': label_logit}