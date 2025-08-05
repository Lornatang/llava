# Copyright (c) AlphaBetter. All rights reserved.
import torch
from transformers.trainer_utils import set_seed

__all__ = [
    "disable_torch_init", "seed_worker",
]


def disable_torch_init() -> None:
    """Disables the reset_parameters method for certain PyTorch layers to prevent unnecessary initialization."""
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def seed_worker(worker_id: int, num_workers: int, rank: int) -> None:
    """Helper function to set worker seed during Dataloader initialization.

    Args:
        worker_id (int): The worker ID.
        num_workers (int): The total number of workers.
        rank (int): The rank of the current process.
    """
    init_seed = torch.initial_seed() % 2 ** 32
    worker_seed = num_workers * rank + init_seed
    set_seed(worker_seed)
