from collections import OrderedDict
from turtle import forward
import torch.nn as nn
from perception3d.models._modules.aggregation.dgcnn_module import DGCNNModule

class DGCNNCls(nn.Module):
    def __init__(self,
            in_channel,
            num_class,
            emb_dims,
            out_dropout,
            k
            ) -> None:
        super().__init__()
        self.encoder = DGCNNModule(in_channel, num_class, emb_dims, out_dropout, k)

    def forward(self, **inputs):
        label_logit = self.encoder(**inputs)
        return {'pred_label_logit': label_logit}

