import torch.nn as nn

from perception3d.core.decorator import named_module


class BaseMetric(object):
    def __call__(self, *, preds, targets):
        raise NotImplementedError