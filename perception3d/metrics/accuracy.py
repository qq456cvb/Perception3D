import numpy as np
from perception3d.core.decorator import named_module
from perception3d.metrics._base import BaseMetric
from scipy.special import softmax
import inspect


class AccuracyMetric(BaseMetric):
    def __init__(self, dim=1) -> None:
        super().__init__()
        self.dim = dim
        
    def __call__(self, *, preds, targets):
        prob = softmax(preds['pred_label_logit'], axis=self.dim)
        return {'metric:acc': np.mean((np.argmax(prob, self.dim) == targets['gt_label']).astype(np.float32))}
    
