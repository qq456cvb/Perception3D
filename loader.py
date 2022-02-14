from omegaconf import OmegaConf
import os
import omegaconf
import yaml
import torch

class EnvTag(object):
    # yaml_tag = u'!ENV'

    def __init__(self, env_var):
        self.env_var = env_var

    def __repr__(self):
        return 'EnvTag({}, contains={})'.format(self.env_var, '')

    @classmethod
    def from_yaml(cls, loader, node):
        print(node)
        print(loader.construct_mapping(node, deep=True))
        return EnvTag(node.value)


def constructor(loader, node):
    # print(node.value)
    return loader.construct_mapping(node, deep=True)
    # @classmethod
    # def to_yaml(cls, dumper, data):
    #     return dumper.represent_scalar(cls.yaml_tag, data.env_var)

# Required for safe_load
yaml.SafeLoader.add_constructor('!ENV', constructor)
# Required for safe_dump
# yaml.SafeDumper.add_multi_representer(EnvTag, EnvTag.to_yaml)

from perceptron3d.ops.ball_query import ball_query
from perceptron3d.ops.interpolate import three_interpolate
from perceptron3d.ops.furthest_point_sample import furthest_point_sample
# from setuptools import sandbox
if __name__ == '__main__':
    print(ball_query(torch.zeros((3, 5, 3), device='cuda'), torch.zeros((3, 3, 3), device='cuda'), 0.1, 256).shape)
    # conf = OmegaConf.load('test.yaml')
    # print(conf)
    # sandbox.run_setup('./setup.py', ['build_ext', '--inplace'])
    # yml = yaml.safe_load(open('test.yaml'))
    # # print(yml)
    # print(yml['c']['pn'])