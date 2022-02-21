import types
import warnings
from omegaconf import OmegaConf
import os
import omegaconf
import yaml
import torch
from ruamel.yaml import YAML
from ruamel.yaml.nodes import (SequenceNode, MappingNode, ScalarNode)
from ruamel.yaml.compat import ordereddict
from ruamel.yaml.comments import (CommentedMap, CommentedOrderedMap, CommentedSet,
                                  CommentedKeySeq, CommentedSeq, TaggedScalar,
                                  CommentedKeyMap)
from ruamel.yaml.constructor import SafeConstructor, DuplicateKeyError, DuplicateKeyFutureWarning, ConstructorError

class CustomConstructor(SafeConstructor):
    def tag_constructor(self, node):
        if isinstance(node, SequenceNode):
            constructor = self.__class__.construct_sequence
            node.tag = 'tag:yaml.org,2002:seq'
        elif isinstance(node, MappingNode):
            constructor = self.__class__.construct_mapping
            node.tag = 'tag:yaml.org,2002:map'
        data = constructor(self, node)
        return data
    
    @classmethod
    def add_pytags(self, tags):
        for tag in tags:
            self.add_constructor(tag, CustomConstructor.tag_constructor)
        
    def flatten_mapping(self, node):
        # type: (Any) -> Any
        """
        This implements the merge key feature http://yaml.org/type/merge.html
        by inserting keys from the merge dict/list of dicts if not yet
        available in this node
        """
        merge = []  # type: List[Any]
        index = 0
        while index < len(node.value):
            key_node, value_node = node.value[index]
            if key_node.tag == u'tag:yaml.org,2002:merge':
                if merge:  # double << key
                    if self.allow_duplicate_keys:
                        del node.value[index]
                        index += 1
                        continue
                    args = [
                        'while constructing a mapping',
                        node.start_mark,
                        'found duplicate key "{}"'.format(key_node.value),
                        key_node.start_mark,
                        """
                        To suppress this check see:
                           http://yaml.readthedocs.io/en/latest/api.html#duplicate-keys
                        """,
                        """\
                        Duplicate keys will become an error in future releases, and are errors
                        by default when using the new API.
                        """,
                    ]
                    if self.allow_duplicate_keys is None:
                        warnings.warn(DuplicateKeyFutureWarning(*args))
                    else:
                        raise DuplicateKeyError(*args)
                del node.value[index]
                print(value_node)
                if isinstance(value_node, MappingNode):
                    self.flatten_mapping(value_node)
                    merge.extend(value_node.value)
                elif isinstance(value_node, SequenceNode):
                    submerge = []
                    for subnode in value_node.value:
                        if not isinstance(subnode, MappingNode):
                            raise ConstructorError(
                                'while constructing a mapping',
                                node.start_mark,
                                'expected a mapping for merging, but found %s' % subnode.id,
                                subnode.start_mark,
                            )
                        self.flatten_mapping(subnode)
                        submerge.append(subnode.value)
                    submerge.reverse()
                    for value in submerge:
                        merge.extend(value)
                else:
                    raise ConstructorError(
                        'while constructing a mapping',
                        node.start_mark,
                        'expected a mapping or list of mappings for merging, '
                        'but found %s' % value_node.id,
                        value_node.start_mark,
                    )
            elif key_node.tag == u'tag:yaml.org,2002:value':
                key_node.tag = u'tag:yaml.org,2002:str'
                index += 1
            else:
                index += 1
        if bool(merge):
            node.merge = merge  # separate merge keys to be able to update without duplicate
            node.value = merge + node.value
            
    def construct_non_recursive_object(self, node, tag=None):
        # type: (Any, Optional[str]) -> Any
        constructor = None  # type: Any
        tag_suffix = None
        if tag is None:
            tag = node.tag
        
        if tag in self.yaml_constructors:
            constructor = self.yaml_constructors[tag]
        else:
            for tag_prefix in self.yaml_multi_constructors:
                if tag.startswith(tag_prefix):
                    tag_suffix = tag[len(tag_prefix) :]
                    constructor = self.yaml_multi_constructors[tag_prefix]
                    break
            else:
                if None in self.yaml_multi_constructors:
                    tag_suffix = tag
                    constructor = self.yaml_multi_constructors[None]
                elif None in self.yaml_constructors:
                    constructor = self.yaml_constructors[None]
                elif isinstance(node, ScalarNode):
                    constructor = self.__class__.construct_scalar
                elif isinstance(node, SequenceNode):
                    constructor = self.__class__.construct_sequence
                elif isinstance(node, MappingNode):
                    constructor = self.__class__.construct_mapping
        if tag_suffix is None:
            data = constructor(self, node)
        else:
            data = constructor(self, tag_suffix, node)
        if isinstance(data, types.GeneratorType):
            generator = data
            data = next(generator)
            if self.deep_construct:
                for _dummy in generator:
                    pass
            else:
                self.state_generators.append(generator)
        return data
            
    # return loader.construct_mapping(node, deep=True)
    # @classmethod
    # def to_yaml(cls, dumper, data):
    #     return dumper.represent_scalar(cls.yaml_tag, data.env_var)

# Required for safe_load
# yaml.SafeLoader.add_constructor('!ENV', constructor)
# Required for safe_dump
# yaml.SafeDumper.add_multi_representer(EnvTag, EnvTag.to_yaml)

from perception3d.ops.ball_query import ball_query
from perception3d.ops.interpolate import three_interpolate
from perception3d.ops.furthest_point_sample import furthest_point_sample
# from setuptools import sandbox
if __name__ == '__main__':
    # print(ball_query(torch.zeros((3, 5, 3), device='cuda'), torch.zeros((3, 3, 3), device='cuda'), 0.1, 256).shape)
    conf = OmegaConf.load('test.yaml')
    print(type(conf))
    print(conf.c.pn.pn)
    # print(conf)
    # conf = yaml.safe_load(open('configs/default.yaml'))
    # print(conf)
    
    # yml = YAML(typ='safe')
    # CustomConstructor.add_pytags(['!torch.utils.data.DataLoader', '!perception3d.datasets.shapenet.ShapeNetPartDataset'])
    # yml.Constructor = CustomConstructor
    # conf = yml.load(open('configs/default.yaml'))
    # print(conf)
    
    # sandbox.run_setup('./setup.py', ['build_ext', '--inplace'])
    # yml = yaml.safe_load(open('test.yaml'))
    # # print(yml)
    # print(yml['c']['pn'])