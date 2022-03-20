from typing import Dict, List
import numpy as np
import torch.distributed as dist
import torch
import pickle


def tensor2numpy(t):
    if isinstance(t, list):
        for i in range(len(t)):
            t[i] = tensor2numpy(t[i])
        return t
    if isinstance(t, dict):
        for k in t:
            t[k] = tensor2numpy(t[k])
        return t
    else:
        return t.detach().cpu().numpy()


def merge_dict(preds, targets, others={}):
    ret = others
    for k in preds:
        ret['_preds' + k] = preds[k]
    for k in targets:
        ret['_targets' + k] = targets[k]
    return ret


def split_dict(dic):
    preds = {}
    targets = {}
    others = {}
    for k in dic:
        if k.startswith('_preds'):
            preds[k[6:]] = dic[k]
        elif k.startswith('_targets'):
            targets[k[8:]] = dic[k]
        else:
            others[k] = dic[k]
    return preds, targets, others
    

# https://github.com/RangiLyu/nanodet/blob/main/nanodet/util/scatter_gather.py
def gather_results_dist(result_part):
    rank = -1
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device="cuda"
    )

    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device="cuda")
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)

    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device="cuda")
    part_send[: shape_tensor[0]] = part_tensor
    part_recv_list = [part_tensor.new_zeros(shape_max) for _ in range(world_size)]

    # gather all result dict
    dist.all_gather(part_recv_list, part_send)

    if rank < 1:
        all_res = {}
        for recv, shape in zip(part_recv_list, shape_list):
            recv_dict = pickle.loads(recv[: shape[0]].cpu().numpy().tobytes())
            for k in recv_dict:
                if k not in all_res:
                    all_res[k] = recv_dict[k]
                else:
                    all_res[k].extend(recv_dict[k])
        return all_res
    
    
def gather_results(outputs: List[Dict], ddp: bool=False):
    results = dict([(k, []) for k in outputs[0]])
    for res in outputs:
        for k in res:
            results[k].append(res[k])
    for k in results:
        try:
            results[k] = np.concatenate(results[k])
        except:
            pass
    all_results = gather_results_dist(results) if ddp else results
    return all_results