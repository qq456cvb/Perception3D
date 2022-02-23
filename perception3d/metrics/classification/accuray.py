from perception3d.core.decorator import named_module
from perception3d.metrics._base import BaseMetric
import torch.nn.functional as F
import torch
import inspect


@named_module
class AccuracyMetric(BaseMetric):
    def __init__(self, dim=1) -> None:
        super().__init__()
        self.dim = dim
        
    def __call__(self, *, preds, targets):
        prob = F.softmax(preds['label_logit'], dim=self.dim)
        return {repr(self): torch.sum(prob.argmax(self.dim) == targets['label']).cpu().numpy()}
    
