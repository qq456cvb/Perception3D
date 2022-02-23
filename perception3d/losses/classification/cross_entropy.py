from turtle import forward
from perception3d.core.decorator import named_module
from perception3d.losses._base import BaseLoss
import torch.nn.functional as F


class CrossEntropyLoss(BaseLoss):
    def __init__(self, *, **kwargs):
        super().__init__()
        self.additional_args = kwargs
        
    def forward(self, *, preds, targets):
        return {'loss_ce': F.cross_entropy(preds['label_logit'], targets['label'], **self.additional_args)}