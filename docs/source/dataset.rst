Dataset
=======
All datasets are subclasses of ``BaseDataset``. 
As loading data may be time consuming, we provide two additional base classes ``MemoryCachedDataset`` and ``DiskCachedDataset`` to accelerate this process.

Any dataset derived from ``MemoryCachedDataset`` will cache the data into memory once it is loaded and never release them. 
``MemoryCachedDataset`` also accepts an optional argument ``preload``, which determines whether to load the whole dataset into memory at the beginning.
Typically, when one needs to do some time-consuming operation on raw data, and these data fit into the memory, ``MemoryCachedDataset`` could be used.

Any dataset derived from ``DiskCachedDataset`` accepts an optional argument ``cache_dir``, and writes data records on disk into ``cache_dir``. 
Typically, when one needs to do some time-consuming operation on raw data, while these data does not fit into the memory, ``DiskCachedDataset`` could be used.

ModelNet
--------
The ModelNet object classification dataset is based on `\"3D ShapeNets: A Deep Representation for Volumetric Shapes\" <https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf>`_, 
which has two versions, containing 10,000+ models from 10 or 40 categories, respectively.

ShapeNet
--------
The ShapeNet part level segmentation dataset is based on `\"A Scalable Active Framework for Region Annotation in 3D Shape Collections\" <http://web.stanford.edu/~ericyi/papers/part_annotation_16_small.pdf>`_, 
containing about 17,000 3D shape point clouds from 16 shape categories. Each category is annotated with 2 to 6 parts, with 50 parts in total.