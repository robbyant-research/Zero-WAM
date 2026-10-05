# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
import torch.distributed as dist

from ..device import get_device
from .logging import logger
from .Simple_Remote_Infer.deploy.websocket_policy_server import WebsocketPolicyServer


class DistributedModelWrapper:
    """
    TODO
    """

    def __init__(self, model, local_rank):
        self.model = model
        self.local_rank = local_rank

    def infer(self, obs):
        return distributed_infer(self.model, obs, self.local_rank)


def distributed_infer(model, obs, local_rank):
    """
    TODO
    """
    rank = dist.get_rank()
    assert rank == local_rank, "distributed_infer can only run at（rank 0)"

    cmd = torch.tensor(1, dtype=torch.int64, device=model.device)
    dist.broadcast(cmd, src=0)

    obj_list = [obs]
    dist.broadcast_object_list(obj_list, src=0)

    result = model.infer(obs)

    return result


def worker_loop(model, local_rank):
    """
    TODO
    """
    device = get_device(local_rank)
    rank = dist.get_rank()

    while True:
        cmd = torch.zeros(1, dtype=torch.int64, device=device)
        dist.broadcast(cmd, src=0)
        cmd_val = cmd.item()

        if cmd_val == -1:
            break
        elif cmd_val == 1:
            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=0)
            obs = obj_list[0]
            _ = model.infer(obs)
        else:
            pass

    logger.info(f"[worker_loop] Rank {rank} exiting.")


def run_async_server_mode(model, local_rank, host, port):
    logger.info("Running in ASYNC SERVER mode")
    if local_rank == 0:
        dist_model = DistributedModelWrapper(model, local_rank=local_rank)
        model_server = WebsocketPolicyServer(dist_model, host=host, port=port)
        model_server.serve_forever()

        cmd = torch.tensor(
            -1,
            dtype=torch.int64,
            device=model.device)
        dist.broadcast(cmd, src=0)
    else:
        try:
            worker_loop(model, local_rank)
        except KeyboardInterrupt:
            logger.info(f"Rank {local_rank}: Shutting down")
