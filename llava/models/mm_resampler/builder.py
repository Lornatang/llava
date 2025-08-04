# Copyright Lornatang. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Dict

import torch
from torch import nn

from .masked_drop import MaskedDrop
from .perceiver import PerceiverResampler
from .qformer import Qformer
from .spatial_pool import SpatialPool

__all__ = [
    "IdentityMap", "build_vision_resampler",
]


class IdentityMap(nn.Module):
    """Identity mapping module for vision projector.

    Methods:
        forward(x, *args, **kwargs): Returns the input as is.
        config: Returns the configuration dictionary.
    """

    def __init__(self) -> None:
        """Initializes the IdentityMap module."""
        super().__init__()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Returns the input tensor unchanged.

        Args:
            x (torch.Tensor): Input tensor.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The input tensor.
        """
        return x

    @property
    def config(self) -> Dict[str, None]:
        """Returns the configuration for the identity projector.

        Returns:
            Dict[str, str]: Configuration dictionary.
        """
        return {"mm_resampler_type": None}


def build_vision_resampler(model_args, **kwargs) -> nn.Module:
    """Builds a vision projector module based on the configuration.

    Args:
        config (Any): Configuration object with projector parameters.
        **kwargs: Additional keyword arguments.

    Returns:
        nn.Module: The constructed vision projector module.

    Raises:
        ValueError: If the projector type is unknown.
    """
    resampler_type = getattr(model_args, "mm_resampler_type", None)
    if resampler_type == "masked_drop":
        return MaskedDrop(model_args)
    elif resampler_type == "spatial_pool":
        return SpatialPool(model_args, **kwargs)
    elif resampler_type == "perceiver":
        return PerceiverResampler(model_args, **kwargs)
    elif resampler_type == "qformer":
        return Qformer(model_args, **kwargs)
    elif resampler_type is None:
        return IdentityMap()

    raise ValueError(f"Unknown resampler type: {resampler_type}")
