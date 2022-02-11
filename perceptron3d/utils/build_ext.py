import subprocess
from setuptools import sandbox
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.core import run_setup

def build_ext(name, src_files):
    print(list(src_files))
    src = ""
    src += "from setuptools import setup\n"
    src += "from torch.utils.cpp_extension import BuildExtension, CUDAExtension\n"
    src += "setup(\n"
    # src += "name='BALL',\n"
    src += "ext_modules=[\n"
    src += f"CUDAExtension('{name}', [\n"
    for fn in src_files:
        src += f"'{fn}',\n"
    src += "])\n"
    src += "],\n"
    src += "zipsafe=False,\n"
    src += "cmdclass={\n"
    src += "'build_ext': BuildExtension\n"
    src += "})\n"
    with open('/tmp/setup.py', 'w') as f:
        f.write(src)
    # subprocess.run(['python', '/tmp/setup.py', 'build_ext', '--inplace'])
    run_setup('/tmp/setup.py', script_args=['build_ext', '--inplace'], stop_after='run')