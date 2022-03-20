import numpy as np


class NormalizeUnitSphere(object):
    def __call__(self, **inputs):
        points = inputs['points']
        points -= np.mean(points, 0)
        points /= np.max(np.linalg.norm(points, -1))
        inputs['points'] = points
        return inputs