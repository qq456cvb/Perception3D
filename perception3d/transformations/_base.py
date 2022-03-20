

import numpy as np


class Identity(object):
    def __init__(self) -> None:
        pass
    
    def __call__(self, **kwargs):
        return kwargs


class Compose(object):
    def __init__(self, transformations) -> None:
        self.transformations = transformations
    
    def call_impl(self, aug, **kwargs):
        res = kwargs
        if isinstance(aug, list):
            for a in aug:
                res = self.call_impl(a, **res)
            return res
        elif isinstance(aug, dict):
            probs, augs = list(zip(*aug.items()))
            if np.sum(probs) < 1. - 1e-7:
                probs.append(1. - np.sum(probs))
                augs.append(Identity())
            aug_chosen = augs[np.random.choice(len(augs), p=probs / np.sum(probs))]
            return self.call_impl(aug_chosen, **res)
        else:
            return aug(**res)
        
    def __call__(self, **kwargs):
        return self.call_impl(self.transformations, **kwargs)