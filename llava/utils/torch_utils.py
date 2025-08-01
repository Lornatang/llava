# Copyright (c) AlphaBetter. All rights reserved.
import torch

__all__ = [
    "disable_torch_init",
]


def disable_torch_init() -> None:
    """Disables the reset_parameters method for certain PyTorch layers to prevent unnecessary initialization."""
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
