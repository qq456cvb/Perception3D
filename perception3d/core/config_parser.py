import types
from typing import Iterable
import warnings
from attr import resolve_types
import hydra
from omegaconf import DictConfig, OmegaConf
from ruamel.yaml import YAML
from ruamel.yaml.nodes import (SequenceNode, MappingNode, ScalarNode)
from ruamel.yaml.compat import Hashable
from ruamel.yaml.constructor import SafeConstructor, DuplicateKeyError, DuplicateKeyFutureWarning, ConstructorError, BaseConstructor
import os
import pathlib
import regex as re
import io
import perception3d
import yaml
import functools
import importlib


# Recursive dictionary merge
# Copyright (C) 2016 Paul Durivage <pauldurivage+github@gmail.com>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import collections

def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.abc.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    for k, v in dct.items():
        if (k in merge_dct and isinstance(merge_dct[k], dict)
                and isinstance(dct[k], collections.abc.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            merge_dct[k] = dct[k]
            
            
class ConfigConstructor(SafeConstructor):
    def tag_constructor(tag, self, node):
        if isinstance(node, SequenceNode):
            constructor = self.__class__.construct_sequence
            node.tag = 'tag:yaml.org,2002:seq'
        elif isinstance(node, MappingNode):
            constructor = self.__class__.construct_mapping
            node.tag = 'tag:yaml.org,2002:map'
        data = constructor(self, node)
        data['__type__'] = tag[1:]
        return data
    
    @classmethod
    def add_pytags(self, tags):
        for tag in tags:
            self.add_constructor(tag, functools.partial(ConfigConstructor.tag_constructor, tag))
        
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
    
    def construct_mapping(self, node, deep=False):
        if isinstance(node, MappingNode):
            self.flatten_mapping(node)
        # type: (Any, bool) -> Any
        """deep is True when creating an object/mapping recursively,
        in that case want the underlying elements available during construction
        """
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None, None, 'expected a mapping node, but found %s' % node.id, node.start_mark
            )
        total_mapping = self.yaml_base_dict_type()
        if getattr(node, 'merge', None) is not None:
            todo = [(node.merge, False), (node.value, False)]
        else:
            todo = [(node.value, True)]
        for values, check in todo:
            mapping = self.yaml_base_dict_type()  # type: Dict[Any, Any]
            for key_node, value_node in values:
                # keys can be list -> deep
                key = self.construct_object(key_node, deep=True)
                # lists are not hashable, but tuples are
                if not isinstance(key, Hashable):
                    if isinstance(key, list):
                        key = tuple(key)
                if not isinstance(key, Hashable):
                    raise ConstructorError(
                        'while constructing a mapping',
                        node.start_mark,
                        'found unhashable key',
                        key_node.start_mark,
                    )

                value = self.construct_object(value_node, deep=True)
                if check:
                    if self.check_mapping_key(node, key_node, mapping, key, value):
                        mapping[key] = value
                else:
                    if key in mapping:
                        dict_merge(mapping[key], value)
                    else:
                        # print(value)
                        mapping[key] = value
            total_mapping.update(mapping)
        return total_mapping
    
class ConfigParser(object):
    def __init__(self) -> None:
        self.tree = {}
        self.file_cache ={}
    
    def _parse_node(self, prefix, file):
        tree = self.tree
        for p in prefix:
            if p not in tree:
                tree[p] = {}
            tree = tree[p]
        tree['__file__'] = file
        
    def _write_tree(self):
        def write(d, depth):
            res = ''
            if '__file__' in d:
                # import pdb; pdb.set_trace()
                content = self.file_cache[d['__file__']].splitlines()
                for l in content:
                    res += ' ' * 2 * depth + l + '\n'
                res += '\n'
            else:
                for k in d:
                    res += ' ' * 2 * depth + k + ':\n'
                    res += write(d[k], depth + 1)
            return res
        return write(self.tree, 0)
    
    def parse_recursive(self, file):
        if isinstance(file, (str, pathlib.Path)):
            file_path = os.path.abspath(file)
            file_path_noext = os.path.splitext(file_path)[0]
            base_path = str(pathlib.Path(os.path.join(os.path.dirname(os.path.abspath(perception3d.__file__)), '../configs')).resolve())
            todos = []
            
            f = io.open(os.path.abspath(file), "r", encoding="utf-8")
            obj = f.read()
            f.close()
            # replace all ${} with mark
            def find_ref(m):
                s = m.group(0)[2:-1]
                if s.startswith('..'):
                    path = os.path.join(file_path_noext, '../' + s[2:].replace('.', '/'))
                elif s.startswith('.'):
                    path = os.path.join(file_path_noext, s[1:].replace('.', '/'))
                else:
                    path = os.path.join(base_path, s.replace('.', '/'))
                
                res = str(pathlib.Path(path).resolve())
                todos.append(res)
                return '${' + os.path.relpath(res, base_path).replace('/', '.') + '}'
            
            res = re.sub(r'(?<!\s*#\s.*)\$\{[\w|\.]+\}', find_ref, obj)
            self.file_cache[file_path] = res
            if file_path_noext not in todos:
                todos.append(file_path_noext)
                
            for path in todos:
                relpath = os.path.relpath(path, base_path)
                locs = relpath.split('/')
                # print(path)
                for i in range(len(locs)):
                    cand_path = os.path.join(base_path, *locs[:i+1])
                    if os.path.exists(cand_path + '.yml') or os.path.exists(cand_path + '.yaml'):
                        yml_path = cand_path + '.yml' if os.path.exists(cand_path + '.yml') else cand_path + '.yaml'
                        self._parse_node(locs[:i+1], yml_path)
                        if file_path != yml_path:
                            self.parse_recursive(yml_path)
        else:
            raise TypeError('unknown file type')
        
    def parse(self, file):
        self.tree = {}
        self.parse_recursive(file)
        res = self._write_tree()
        
        def traverse(tree, depth):
            if isinstance(tree, DictConfig):
                for k in tree:
                    traverse(tree[k], depth + 1)
                    tree[k] = tree[k]
        # replace << with <<<
        res = re.sub(r'(?<!\s*#\s.*)<<\s*:', '__<<<__:', res)
        yml = YAML(typ='safe')
        ConfigConstructor.add_pytags(['!torch.utils.data.DataLoader', '!perception3d.datasets.shapenet.ShapeNetPartDataset'])
        yml.Constructor = ConfigConstructor
        # load with custom tags
        conf = yml.load(res)
        # use omegaconf to resolve interpolations
        conf = OmegaConf.create(conf)
        traverse(conf, 0)
        conf = OmegaConf.to_yaml(conf)
        
        # convert merge tag back
        conf = re.sub(r'(?<!\s*#\s.*)__<<<__\s*:', '<<:', conf)
        # load again
        conf = yml.load(conf)
        
        # def init_pyinstance(tree, par, par_k):
        #     if isinstance(tree, dict):
        #         for k in tree:
        #             init_pyinstance(tree[k], tree, k)
        #         if '__type__' in tree:
        #             params = dict([(k, v) for k, v in tree.items() if k != '__type__'])
        #             module_name, class_name = tree['__type__'].rsplit(".", 1)
        #             par[par_k] = getattr(importlib.import_module(module_name), class_name)(**params)
        
        # init_pyinstance(conf, None, '')
        return conf


if __name__ == '__main__':
    parser = ConfigParser()
    res = parser.parse('configs/default.yaml')
    print(res)
    # replace << with <<<
    # res = re.sub(r'(?<!\s*#\s.*)<<\s*:', '<<<:', res)
    # yml = YAML(typ='safe')
    # ConfigConstructor.add_pytags(['!torch.utils.data.DataLoader', '!perception3d.datasets.shapenet.ShapeNetPartDataset'])
    # yml.Constructor = ConfigConstructor
    # conf = yml.load(res)
    # open('test.yaml', 'w').write(res)
    # conf = OmegaConf.create(conf)
    # traverse(conf, 0)
    # conf = OmegaConf.to_yaml(conf)
    # conf = re.sub(r'(?<!\s*#\s.*)<<<\s*:', '<<:', conf)
    # conf = yml.load(conf)
    # print(conf)
    # print(OmegaConf.load('test.yaml'))