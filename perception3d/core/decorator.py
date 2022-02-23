
import functools
import inspect
import os

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