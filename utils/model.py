"""
@author: Xiaoping Yue
"""

import torch
from torch import distributed as dist


def load_model(model_type, device, **kwargs):
    if model_type == 'Transformer':
        from model.transformer import build_vanilla_transformer
        model = build_vanilla_transformer(**kwargs)
    elif model_type == 'DWAM-Former':
        from model.dwamformer import DWAMFormer
        model = DWAMFormer(**kwargs)
    else:
        raise KeyError(f'Unknown model type: {model_type}')

    if device == 'cuda':
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()], find_unused_parameters=True)

    return model
