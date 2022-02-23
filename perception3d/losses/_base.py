import torch.nn as nn

class BaseLoss(nn.Module):
    def forward(self, *, preds, targets):
        raise NotImplementedError