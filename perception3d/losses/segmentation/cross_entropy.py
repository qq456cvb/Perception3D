from turtle import forward
from perception3d.core.decorator import named_module
from perception3d.losses._base import BaseLoss
from perception3d.losses.classification.cross_entropy import CrossEntropyLoss as ClsCrossEntropyLoss
import torch.nn.functional as F


CrossEntropyLoss = ClsCrossEntropyLoss