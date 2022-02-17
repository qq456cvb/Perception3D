from perception3d.augmentations.base import Compose
import os
import torch


class BaseDataset(object):
    def __init__(self, *, **kwargs) -> None:
        self.transform = Compose(kwargs['augmentations'])
    
    def get_item(self, idx):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        item = self.get_item(idx)
        
        item_post_aug = self.transform(item)
        return item_post_aug
    

class CachedDataset(BaseDataset):
    def __init__(self, *, **kwargs):
        super().__init__(**kwargs)
    
    def __get_cache__(self, idx):
        raise NotImplementedError
    
    def __put_cache__(self, idx, item):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        item_cached = self.__get_cache__(idx)
        if item_cached is not None:
            item = item_cached
        else:
            item = self.get_item(idx)
            self.__put_cache__(idx, item)
        
        item_post_aug = self.transform(item)
        return item_post_aug
    

class DiskCachedDataset(CachedDataset):
    def __init__(self, *, **kwargs):
        super().__init__(**kwargs)
        self.root = kwargs['cache_dir'] if 'cache_dir' in kwargs else os.path.join(os.getcwd(), 'cache_dir')
        
    def __get_cache__(self, idx):
        path = os.path.join(self.root, '{}.th')
        if os.path.exists(path):
            return torch.load(path)
        else:
            return None
    
    def __put_cache__(self, idx, item):
        path = os.path.join(self.root, '{}.th')
        if not os.path.exists(path):
            torch.save(item, path)
            

class MemoryCachedDataset(CachedDataset):
    def __init__(self, *, **kwargs):
        super().__init__(**kwargs)
        if 'preload' not in kwargs or kwargs['preload'] == False:
            self.caches = [None for _ in range(len(self))]
        else:
            self.caches = [self.get_item(idx) for idx in range(len(self))]
        
    def __get_cache__(self, idx):
        if self.caches[idx] is not None:
            return self.caches[idx]
        else:
            return None
    
    def __put_cache__(self, idx, item):
        if self.caches[idx] is None:
            self.caches[idx] = item