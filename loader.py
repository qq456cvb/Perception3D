from omegaconf import OmegaConf
import os
import yaml

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
    return loader.construct_mapping(node, deep=True)
    # @classmethod
    # def to_yaml(cls, dumper, data):
    #     return dumper.represent_scalar(cls.yaml_tag, data.env_var)

# Required for safe_load
yaml.SafeLoader.add_constructor('!ENV', constructor)
# Required for safe_dump
# yaml.SafeDumper.add_multi_representer(EnvTag, EnvTag.to_yaml)

if __name__ == '__main__':
    yml = yaml.safe_load(open('test.yaml'))
    # print(yml)
    print(yml['c']['pn'])