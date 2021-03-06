import os
import tempfile
from distutils.core import run_setup
import re
from glob import glob


def build_ext(name, working_dir):
    src_files = list(filter(lambda fn: re.match(r'.+\.(cpp|cc|cu)', fn), glob(os.path.join(working_dir, 'src/*'))))
    src = ""
    src += "from setuptools import setup\n"
    src += "from torch.utils.cpp_extension import BuildExtension, CUDAExtension\n"
    src += "setup(\n"
    src += "ext_modules=[\n"
    src += f"CUDAExtension('{name}', [\n"
    for fn in src_files:
        fn = fn.replace('\\', '/')
        src += f"'{fn}',\n"
    src += "])\n"
    src += "],\n"
    src += "zipsafe=False,\n"
    src += "cmdclass={\n"
    src += "'build_ext': BuildExtension\n"
    src += "})\n"
    _, setup_tmpfile = tempfile.mkstemp(suffix='.py')
    with open(setup_tmpfile, 'w') as f:
        f.write(src)
    
    print(src)
    if working_dir is not None:
        wd = os.getcwd()
        os.chdir(working_dir)
    run_setup(setup_tmpfile, script_args=['build_ext', '--inplace'], stop_after='run')
    if working_dir is not None:
        os.chdir(wd)