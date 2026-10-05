# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.distributed as dist

from ..device import distributed_backend, get_device, set_device


def _configure_model(model, shard_fn, param_dtype, device, eval_mode=True):
    """
    TODO
    """
    if eval_mode:
        model.eval().requires_grad_(False)
    if dist.is_initialized():
        dist.barrier()

    if dist.is_initialized():
        model = shard_fn(model)
    else:
        model.to(param_dtype)
        model.to(device)

    return model


def init_distributed(world_size, local_rank, rank, requested_device=None):
    device = get_device(local_rank, requested_device)
    set_device(device)
    kwargs = {
        "backend": distributed_backend(device),
        "init_method": "env://",
        "rank": rank,
        "world_size": world_size,
    }
    if device.type != "cpu":
        kwargs["device_id"] = device
    dist.init_process_group(**kwargs)

def dist_mean(local_tensor):
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.AVG)
    return local_tensor

def dist_max(local_tensor):
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.MAX)
    return local_tensor
