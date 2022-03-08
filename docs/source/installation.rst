Installation
============


Prerequisites
-------------
Perception3D is tested with Pytorch>=1.3.1, CUDA>=10.0, Pytorch Lightning>=1.4.1.

You can install them with the following command:

.. code-block:: rst

    pip install torch==1.10.2+cu102 torchvision==0.11.3+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
    conda install protobuf
    pip install ruamel.yaml==0.16.13 omegaconf==2.0.5 pytorch-lightning==1.5.10

Demo
------
After installing the prerequisites, then you are ready to go! You can directly run ``run.py`` under the root directory of Perception3D.