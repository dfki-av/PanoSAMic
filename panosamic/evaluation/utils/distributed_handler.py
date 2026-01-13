"""
Author: Mahdi Chamseddine
"""

import os
from pathlib import Path
from typing import TypeVar

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

T = TypeVar(
    "T",
    torch.Tensor,
    list[torch.Tensor],
    dict[str, torch.Tensor],
    list[dict[str, torch.Tensor]],
)


class DistributedHandler:
    def __init__(self, n_gpus: int, backend: str = "nccl") -> None:
        # Check whether we are operating with more than 1 GPU.
        if dist.is_available() and n_gpus > 1:
            dist.init_process_group(backend)
        # Ensure that DataLoader workers can also be started correctly in multi-GPU
        # scenarios.
        mp.set_start_method("forkserver")

        # Check whether we have CUDA support.
        self.has_cuda = torch.cuda.is_available()
        # Check whether are running a multi-GPU job.
        self.is_distributed = dist.is_initialized()
        # Total number of main processes / GPUs.
        self.world_size = int(os.environ["WORLD_SIZE"]) if self.is_distributed else 1
        # Global ID (across nodes) of this process.
        self.rank = int(os.environ["RANK"]) if self.is_distributed else 0
        # Local ID (on this node) of this process. This corresponds to the ID of the GPU
        # on the current node that this process is responsible for.
        self.local_rank = int(os.environ["LOCAL_RANK"]) if self.is_distributed else 0
        # Things that should be done only once (writing logs, checkpoints, ...) should
        # be done only by the master process.
        self.is_master = self.rank == 0

    def load_state(self, check_point_path: Path):
        if self.has_cuda:
            map_location = f"cuda:{self.local_rank:d}"
        else:
            map_location = "cpu"
        state = torch.load(
            check_point_path, map_location=map_location, weights_only=True
        )
        return state

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Prepares the model for multi-GPU training if necessary."""
        if self.is_distributed:
            device_ids = [self.local_rank] if self.has_cuda else None
            model = DistributedDataParallel(
                model, device_ids=device_ids, output_device=self.local_rank
            )
        return model

    def move_to_gpu(self, data: T, keys: list[str] | None = None) -> T:
        """Moves data to the GPU, can handle tensors, list of tensors, dict of tensors,
        and list of dict of tensors"""
        if self.has_cuda:
            if isinstance(data, torch.Tensor):
                data = data.to(self.local_rank)
            elif isinstance(data, dict):
                if keys is None:
                    keys = list(data.keys())
                for key in keys:
                    data[key] = self.move_to_gpu(data[key], None)
            elif isinstance(data, list):
                data = [self.move_to_gpu(item, keys) for item in data]  # type:ignore
            else:
                raise NotImplementedError("Input type unknown.")
        return data

    def move_to_cpu(self, data: T, keys: list[str] | None = None) -> T:
        """Moves data to the CPU, can handle tensors, list of tensors, dict of tensors,
        and list of dict of tensors"""
        if isinstance(data, torch.Tensor):
            data = data.cpu()
        elif isinstance(data, dict):
            if keys is None:
                keys = list(data.keys())
            for key in keys:
                data[key] = self.move_to_cpu(data[key], None)
        elif isinstance(data, list):
            data = [self.move_to_cpu(item, keys) for item in data]  # type:ignore
        else:
            raise NotImplementedError("Input type unknown.")
        return data

    def merge_distributed_values(
        self,
        values: torch.Tensor | dict[str, torch.Tensor],
        reduction: dist.ReduceOp.RedOpType = dist.ReduceOp.AVG,
    ) -> None:
        """
        Combines the values of the tensors in the dictionary from all devices with the
        specified reduction method and makes result available in the respective tensor
        on rank 0 (master).
        """
        if not self.is_distributed:
            return
        if isinstance(values, torch.Tensor):
            dist.reduce(values, 0, reduction)
        else:
            for key in values.keys():
                dist.reduce(values[key], 0, reduction)

    def broadcast_value(self, value: torch.Tensor) -> None:
        """Broadcasts a value from the main process to other processes"""
        if not self.is_distributed:
            return
        dist.broadcast(value, 0)

    def print(self, message: str) -> None:
        """Prints the given string only on rank 0 (master) to avoid multiple print-outs
        of the message."""
        if self.is_master:
            print(message)

    def close(self):
        if self.is_distributed:
            self.print("Closing process group")
            dist.destroy_process_group()
