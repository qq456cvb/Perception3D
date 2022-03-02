Configuration System
====================

Learn about configs
-------------------

Perception3D disentangles code into static Python class definitions and dynamic runtime configurations.

Static Python class definitions are the standard Python code that defines various models, datasets, etc. They are placed under the folder ``perception3d/``.

Dynamic runtime configurations define the necessary arguments for the Python class when running a model. They are placed under the folder ``configs/``.

Configuration in YAML format with custom syntax
-----------------------------------------------
The dynamic configuration files are in `YAML <https://learnxinyminutes.com/docs/yaml/>`_  format, with some additional syntax provided by Perception3D.

Let's look at the following example located at ``configs/run/obj_classification.yaml``:

.. code-block:: rst
    :linenos:

    default_cfg:
        max_epochs: 100 
        gpus: [0]
        accelerator: 'gpu'
        strategy: 'ddp'
        log_dir: 'shapenet_log'
        dataloader: !torch.utils.data.DataLoader
            <<: ${.dataloader_cfg}
    
    dataloader_cfg:
        batch_size: 16
        num_workers: 10
        dataset: !perception3d.datasets.shapenet.ShapeNetPartDataset
            <<: ${datasets.shapenet}

In line 1, we define a configuration named ``default_cfg`` with the following keys: ``max_epochs``, ``gpus``, etc. 
The tag ``!torch.utils.data.DataLoader`` in line 7 defines a special configuration, meaning that ``dataloader`` is an object instance of Python class ``torch.utils.data.DataLoader``.
Line 8 defines the keyword arguments that would pass into the contructor of ``torch.utils.data.DataLoader``, where ``<<`` is the `YAML <https://learnxinyminutes.com/docs/yaml/>`_ merge syntax, flattening the dictionary arguments.

.. note:: 
    Perception3D would automatically create object instance of Python class named after ``!`` syntax, and you should make sure that this name is valid and can be imported from Python.

``${.dataloader_cfg}`` in line 8 represents a configuration reference, and any reference should be surrounded by ``${}``. Perception3D uses a similar syntax and parsing system with Python import system.
``.dataloader_cfg`` is a relative reference and refers to the dictionary named ``dataloader_cfg`` in the same file (in this case, line 10).
One could also specifiy a absolute reference, as shown in line 14, where ``datasets.shapenet`` refers to the configuration located under ``configs/datasets/shapenet.yaml``.

.. note:: 
    Perception3D automatically determines the type of reference, which can be a YAML file ((e.g., ``datasets.shapenet`` -> ``configs/datasets/shapenet.yaml``), or a variable in the YAML file (e.g., ``datasets.shapenet.num_sample`` -> ``num_sample`` variable in ``configs/datasets/shapenet.yaml``), and you do not need to explicitly specify the ``.yaml`` extension.

