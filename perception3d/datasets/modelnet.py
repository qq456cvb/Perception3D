from ntpath import join
import os
import numpy as np
from perception3d.datasets._base import BaseDataset, MemoryCachedDataset, download
from glob import glob
import shutil
from perception3d.utils.reader import read_off, sample_vertex_from_mesh


class ModelNetClsDataset(MemoryCachedDataset):
    def __init__(self, *, root, nclasses, split, num_sample, **kwargs):
        assert nclasses in [10, 40]
        if nclasses == 10:
            self.dataset_url = 'https://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
        else:
            self.dataset_url = 'https://modelnet.cs.princeton.edu/ModelNet40.zip'
            
        self.root = root
        self.split = split
        self.num_sample = num_sample

        if not os.path.exists(self.root) or not list(filter(lambda fn: fn[-4:] != '.zip', os.listdir(self.root))):
            print('Downloading ModelNet data files...')
            os.makedirs(self.root, exist_ok=True)
            download(self.dataset_url, self.root)
            
            print('Preprocessing files...')
            data_path = os.path.join(self.root, f'ModelNet{nclasses}')
            for d in os.listdir(data_path):
                if os.path.isdir(os.path.join(data_path, d)):
                    shutil.move(os.path.join(data_path, d), self.root)
            shutil.rmtree(data_path)
            if nclasses == 10:
                shutil.rmtree(os.path.join(self.root, '__MACOSX'))

        self.valid_names = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        
        self.records = []
        self.classes = []
        for i, name in enumerate(self.valid_names):
            records = list(glob(os.path.join(self.root, name, split, '*.off')))
            self.records.extend(records)
            self.classes.extend([i] * len(records))
        super().__init__(**kwargs)
        
    def __len__(self):
        return len(self.records)
    
    def get_item(self, idx):
        fn = self.records[idx]
        pts, faces = read_off(open(fn))
        pts = sample_vertex_from_mesh(pts, faces, num_samples=self.num_sample)[0]
        label = self.classes[idx]
        
        return {'points': pts.astype(np.float32), 'gt_label': label}
        
        
if __name__ == '__main__':
    ds = ModelNetClsDataset(root='modelnet10', nclasses=10, split='test', num_sample=2048)
    for d in ds:
        print(d['points'].shape, d['gt_label'])