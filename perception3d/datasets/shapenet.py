import os
import numpy as np
from perception3d.datasets._base import BaseDataset, MemoryCachedDataset, download
from glob import glob

id2name = {
    '04379243': 'table',
    '03593526': 'jar',
    '04225987': 'skateboard',
    '02958343': 'car',
    '02876657': 'bottle',
    '04460130': 'tower',
    '03001627': 'chair',
    '02871439': 'bookshelf',
    '02942699': 'camera',
    '02691156': 'airplane',
    '03642806': 'laptop',
    '02801938': 'basket',
    '04256520': 'sofa',
    '03624134': 'knife',
    '02946921': 'can',
    '04090263': 'rifle',
    '04468005': 'train',
    '03938244': 'pillow',
    '03636649': 'lamp',
    '02747177': 'trash bin',
    '03710193': 'mailbox',
    '04530566': 'watercraft',
    '03790512': 'motorbike',
    '03207941': 'dishwasher',
    '02828884': 'bench',
    '03948459': 'pistol',
    '04099429': 'rocket',
    '03691459': 'loudspeaker',
    '03337140': 'file cabinet',
    '02773838': 'bag',
    '02933112': 'cabinet',
    '02818832': 'bed',
    '02843684': 'birdhouse',
    '03211117': 'display',
    '03928116': 'piano',
    '03261776': 'earphone',
    '04401088': 'telephone',
    '04330267': 'stove',
    '03759954': 'microphone',
    '02924116': 'bus',
    '03797390': 'mug',
    '04074963': 'remote',
    '02808440': 'bathtub',
    '02880940': 'bowl',
    '03085013': 'keyboard',
    '03467517': 'guitar',
    '04554684': 'washer',
    '02834778': 'bicycle',
    '03325088': 'faucet',
    '04004475': 'printer',
    '02954340': 'cap',
}

name2id = dict([(v, k) for k, v in id2name.items()])


class ShapeNetPartDataset(MemoryCachedDataset):
    valid_names = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife',
                   'lamp', 'laptop', 'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
    dataset_url = 'https://shapenet.cs.stanford.edu/iccv17/partseg'
    
    def __init__(self, *, root, split, num_sample, **kwargs) -> None:
        self.root = root
        self.split = split
        self.num_sample = num_sample
        self.data_path = os.path.join(root, split)
        if not os.path.exists(self.data_path) or not list(filter(lambda fn: fn[-4:] != '.zip', os.listdir(self.data_path))):
            print('Downloading ShapeNet data files...')
            os.makedirs(self.data_path, exist_ok=True)
            download(f'{self.dataset_url}/{split}_data.zip', self.data_path)
            download(f'{self.dataset_url}/{split}_label.zip', self.data_path)
            
        self.records = []
        self.classes = []
        for name in self.valid_names:
            synset_id = name2id[name]
            records = list(glob(os.path.join(self.data_path, f'{split}_data', synset_id, '*.pts')))
            self.records.extend(records)
            self.classes.extend([self.valid_names.index(name)] * len(records))
        super().__init__(**kwargs)
        
    def __len__(self):
        return len(self.records)
    
    def get_item(self, idx):
        def rreplace(s, old, new):
            return new.join(s.rsplit(old, 1))
        fn = self.records[idx]
        pts = np.loadtxt(fn)
        labels = np.loadtxt(rreplace(rreplace(fn, 'pts', 'seg'), '_data', '_label'))
        
        sample_idx = np.random.randint(pts.shape[0], size=(self.num_sample,))
        pts = pts[sample_idx]
        labels = labels[sample_idx]
        
        label_cls = self.classes[idx]
        return {'points': pts.astype(np.float32), 'gt_label': labels.astype(np.int64), 'gt_label_cls': label_cls}


if __name__ == '__main__':
    ds = ShapeNetPartDataset(root='shapenet', split='test', num_sample=2048)
    for d in ds:
        print(d['points'].shape, d['gt_label'].shape)