
import functools
import inspect
import os
from typing import Hashable
from frozendict import frozendict


def freezeargs(func):
    """Transform mutable dictionnary
    Into immutable
    Useful to be compatible with cache
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple([tuple(x) if isinstance(x, list) else x for x in args])
        args = tuple([frozendict(arg) if isinstance(arg, dict) else arg for arg in args])
        kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in kwargs.items()}
        kwargs = {k: frozendict(v) if isinstance(v, dict) else v for k, v in kwargs.items()}
        
        # print(args)
        print(len(args), type(args[0]), type(args[1]))
        return func(*args, **kwargs)
    return wrapped


def named_module(cls):
    cls.__cnt__ = -1
    old_init = cls.__init__
    def new_init(*args, **kwargs):
        old_init(*args, **kwargs)
        cls.__cnt__ += 1
    cls.__init__ = new_init
    fn = inspect.getfile(cls)
    cls.__repr__ = lambda self: os.path.splitext(fn[fn.rfind('perception3d') + len('perception3d') + 1:])[0].replace('/', '.') + '.' + cls.__name__ + '.' + str(cls.__cnt__)
    return cls