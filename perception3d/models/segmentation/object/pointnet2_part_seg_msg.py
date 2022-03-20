from collections import OrderedDict
from typing import List
from perception3d.models.classification.object.pointnet2_cls_msg import PointNet2ClsMSG
from perception3d.models._modules.aggregation.point_sa_module import PointSAModuleMSG
from perception3d.models._modules.propagation.point_fp_module import PointFPModule
import torch.nn as nn
import torch
import torch.nn.functional as F
class PointNet2SegMSG(PointNet2ClsMSG):
    def __init__(self,
                 in_channels,
                 num_class,
                 num_seg_class,
                 sa_modules: List[PointSAModuleMSG] = [],
                 fp_modules: List[PointFPModule] = [],
                 out_mlp_channels=[],
                 out_dropouts=[],
                 bn=True
                 ):
        super().__init__(in_channels=in_channels, num_class=num_seg_class, sa_modules=sa_modules, out_mlp_channels=out_mlp_channels, out_dropouts=out_dropouts, bn=bn)
        self.num_class = num_class
        self.num_seg_class = num_seg_class
        self.FP_modules = nn.ModuleList()
        ch = self.feature_channels[len(fp_modules)]
        for i in range(len(fp_modules)):
            last_ch = self.feature_channels[len(fp_modules) - i - 1]
            fp_modules[i].mlps = nn.Sequential(
                nn.Conv2d(
                    ch + last_ch + (self.num_class if i == len(fp_modules) - 1 else 0),
                    fp_modules[i].mlp_channels[0],
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    bias=False if bn else True),
                fp_modules[i].mlps
            )
            ch = fp_modules[i].mlp_channels[-1]
            self.FP_modules.append(fp_modules[i])
        self.num_fp = len(self.FP_modules)
        
        self.out_mlps = nn.ModuleList()
        for i in range(len(out_mlp_channels) - 1):
            mlp = nn.Sequential()
            mlp.add_module(f'layer{i}',
                nn.Conv1d(out_mlp_channels[i], out_mlp_channels[i + 1], 1))
            if bn:
                mlp.add_module(f'layer{i}_bn', nn.BatchNorm1d(out_mlp_channels[i + 1]))
            mlp.add_module(f'layer{i}_relu', nn.ReLU())
            if out_dropouts[i] > 0:
                mlp.add_module(f'layer{i}_dropout', nn.Dropout(out_dropouts[i]))
            self.out_mlps.append(mlp)

        self.out_mlps.append(nn.Sequential(OrderedDict([(f'layer{len(out_mlp_channels) - 1}', nn.Conv1d(out_mlp_channels[-1], num_seg_class, 1))])))
        
    def forward(self, **inputs):
        xyz = inputs['points']
        if 'features' in inputs:
            features = inputs['features']
        else:
            features = None
        
        sa_xyzs = [xyz]
        sa_features = [features]
        assert 'gt_label_cls' in inputs
        label_cls = inputs['gt_label_cls']
        for i in range(self.num_sa):
            xyz, features, _ = self.SA_modules[i](sa_xyzs[-1], sa_features[-1])
            sa_xyzs.append(xyz)
            sa_features.append(features)
            
        for i in range(self.num_fp):
            reverse_i = self.num_fp - i
            if i != self.num_fp - 1:
                sa_features[reverse_i - 1] = self.FP_modules[i](sa_xyzs[reverse_i - 1], sa_xyzs[reverse_i], sa_features[reverse_i - 1], sa_features[reverse_i])
            else:
                features = torch.cat([F.one_hot(label_cls, self.num_class)[..., None].expand(-1, -1, sa_xyzs[0].shape[1]), sa_xyzs[0].transpose(1, 2)], dim=1)
                if sa_features[0] is not None:
                    features = torch.cat([features, sa_features[0]], dim=1)
                sa_features[0] = self.FP_modules[i](sa_xyzs[0], sa_xyzs[1], features, sa_features[1])
        
        features = sa_features[0]
        for i in range(len(self.out_mlps)):
            features = self.out_mlps[i](features)
        return {'pred_label_logit': features}
            
                