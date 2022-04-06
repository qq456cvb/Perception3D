Loss
====
Loss defines the training target, and derives from ``BaseLoss``. 
It has a ``forward`` function which accepts two keyword arguments: ``preds`` and ``targets``, and is invoked for every training step.
``preds`` is the output from network models, which is a Python dictionary. ``targets`` is the output of datasets, which is a Python dictionary too.
The ``forward`` function should return a Python dictionary, where keys are the name for different losses and values are loss tensors.

.. note::
    Perception3D will ensure that ``preds`` and ``targets`` both contain **batch** collated **PyTorch** tensors, and the output loss should be **PyTorch** tensors too.
