from perception3d.core.decorator import named_module
from perception3d.metrics._base import BaseMetric
import torch.nn.functional as F
import torch
import inspect


class AccuracyMetric(BaseMetric):
    def __init__(self, dim=1) -> None:
        super().__init__()
        self.dim = dim
        
    def __call__(self, *, preds, targets):
        prob = F.softmax(preds['pred_label_logit'], dim=self.dim)
        return {'metric_acc': torch.sum(prob.argmax(self.dim) == targets['gt_label'])}
    
