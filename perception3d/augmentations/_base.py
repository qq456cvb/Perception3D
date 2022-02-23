

import numpy as np


class Identity(object):
    def __init__(self) -> None:
        pass
    
    def __call__(self, **kwargs):
        return kwargs


class Compose(object):
    def __init__(self, augmentations) -> None:
        self.augmentations = augmentations
    
    def call_impl(self, aug, **kwargs):
        res = kwargs
        if isinstance(aug, list):
            for a in aug:
                res = self.call_impl(a, **res)
            return res
        elif isinstance(aug, dict):
            probs, augs = list(zip(*aug.items()))
            aug_chosen = augs[np.random.choice(len(augs), p=probs / np.sum(probs))]
            return self.call_impl(aug_chosen, **res)
        elif aug is None:
            return res
        else:
            return self.call_impl(aug, **res)
        
    def __call__(self, **kwargs):
        return self.call_impl(self.augmentations, **kwargs)