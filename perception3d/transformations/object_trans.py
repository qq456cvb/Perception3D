import numpy as np
from perception3d.ops.furthest_point_sample import furthest_point_sample_nocuda
import torch
from perception3d.utils.reader import sample_vertex_from_mesh


class NormalizeUnitSphere(object):
    def __call__(self, **inputs):
        points = inputs['points']
        points -= np.mean(points, 0)
        points /= np.max(np.linalg.norm(points, axis=-1))
        inputs['points'] = points
        return inputs

class SampleMeshUniform(object):
    def __init__(self, num_sample) -> None:
        self.num_sample = num_sample
        
    def __call__(self, **inputs):
        vertexs = inputs['vertexs']
        faces = inputs['faces']
        points = sample_vertex_from_mesh(vertexs, faces, num_samples=self.num_sample)[0]
        inputs['points'] = points
        del inputs['vertexs']
        del inputs['faces']
        return inputs


class FurthestPointSample(object):
    def __init__(self, num_sample) -> None:
        self.num_sample = num_sample
        
    def __call__(self, **inputs):
        points = inputs['points']
        idx = furthest_point_sample_nocuda(torch.from_numpy(points[None]), self.num_sample)[0].numpy()
        inputs['points'] = points[idx]
        return inputs