from urllib.parse import urlparse
from perception3d.transformations._base import Compose, Identity
import os
import torch
import sys
from contextlib import closing
from shutil import copyfileobj
import urllib.request as request
import zipfile
import requests


from tqdm import tqdm


def download_file(url: str, fname: str):
    resp = requests.get(url, stream=True, verify=False)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
            
            
def download(url, save_dir, unzip=True, delete_after=True):
    namespace = sys._getframe(1).f_globals  # caller's globals
    c_path = os.path.dirname(namespace['__file__'])
    fn = os.path.basename(urlparse(url).path)
    fn = fn if fn.strip() else "tmp.dl"
    dl_path = os.path.join(save_dir, fn)
    download_file(url, dl_path)
    folder = os.path.splitext(fn)[0]
    save_dir = save_dir if save_dir.strip() else os.path.join(c_path, folder)
    if unzip:
        with zipfile.ZipFile(dl_path, 'r') as zip_ref:
            zip_ref.extractall(save_dir)   
    if delete_after and unzip and os.path.isfile(dl_path):
        os.remove(dl_path)

class BaseDataset(object):
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
        self.transform = Compose(kwargs['transformations']) if 'transformations' in kwargs else Identity()
    
    def get_item(self, idx):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        item = self.get_item(idx)
        
        item_post_aug = self.transform(item)
        return item_post_aug
    

class CachedDataset(BaseDataset):
    def __init__(self, **kwargs):
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
        
        item_post_aug = self.transform(**item)
        return item_post_aug
    

class DiskCachedDataset(CachedDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cache_root = kwargs['cache_dir'] if 'cache_dir' in kwargs else os.path.join(os.getcwd(), 'cache_dir')
        
    def __get_cache__(self, idx):
        path = os.path.join(self._cache_root, '{}.th')
        if os.path.exists(path):
            return torch.load(path)
        else:
            return None
    
    def __put_cache__(self, idx, item):
        path = os.path.join(self._cache_root, '{}.th')
        if not os.path.exists(path):
            torch.save(item, path)
            

class MemoryCachedDataset(CachedDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'preload' not in kwargs or kwargs['preload'] == False:
            self._caches = [None for _ in range(len(self))]
        else:
            self._caches = [self.get_item(idx) for idx in range(len(self))]
        
    def __get_cache__(self, idx):
        if self._caches[idx] is not None:
            return self._caches[idx]
        else:
            return None
    
    def __put_cache__(self, idx, item):
        if self._caches[idx] is None:
            self._caches[idx] = item