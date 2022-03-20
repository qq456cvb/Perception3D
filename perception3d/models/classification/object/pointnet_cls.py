from collections import OrderedDict
import torch.nn as nn

from perception3d.models._modules.aggregation.pointnet_module import PointNetModule


class PointNetCls(nn.Module):
    def __init__(self,
                 in_channel,
                 num_class,                 
                 out_mlp_channels=[],
                 out_dropouts=[],
                 bn=True) -> None:
        super().__init__()
        self.out_mlps = nn.ModuleList()
        for i in range(len(out_mlp_channels) - 1):
            mlp = nn.Sequential()
            mlp.add_module(f'layer{i}',
                           nn.Linear(out_mlp_channels[i], out_mlp_channels[i + 1]))
            if bn:
                mlp.add_module(f'layer{i}_bn', nn.BatchNorm1d(
                    out_mlp_channels[i + 1]))
            mlp.add_module(f'layer{i}_relu', nn.ReLU())
            if out_dropouts[i] > 0:
                mlp.add_module(f'layer{i}_dropout', nn.Dropout(out_dropouts[i]))
            self.out_mlps.append(mlp)

        self.out_mlps.append(nn.Sequential(OrderedDict(
            [(f'layer{len(out_mlp_channels) - 1}', nn.Linear(out_mlp_channels[-1], num_class))])))
        
        self.encoder = PointNetModule(global_feat=True, feature_transform=True, channel=in_channel)
    
    def forward(self, **inputs):
        features = self.encoder(**inputs)[0]
        for mlp in self.out_mlps:
            features = mlp(features)
        label_logit = features
        return {'pred_label_logit': label_logit}