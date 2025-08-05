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
import re
from typing import Any, Dict

import torch
from torch import nn

__all__ = [
    "IdentityMap", "SimpleResBlock", "build_vision_projector"
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
    def config(self) -> Dict[str, str]:
        """Returns the configuration for the identity projector.

        Returns:
            Dict[str, str]: Configuration dictionary.
        """
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    """Simple residual block with LayerNorm and MLP.

    Args:
        channels (int): Number of input and output channels.

    Methods:
        forward(x): Applies normalization and residual MLP.
    """

    def __init__(self, channels: int) -> None:
        """Initializes the SimpleResBlock.

        Args:
            channels (int): Number of channels for input and output.
        """
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies layer normalization and residual MLP to the input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after residual connection.
        """
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config: Any, delay_load: bool = False, **kwargs) -> nn.Module:
    """Builds a vision projector module based on the configuration.

    Args:
        config (Any): Configuration object with projector parameters.
        delay_load (bool, optional): Whether to delay loading. Defaults to False.
        **kwargs: Additional keyword arguments.

    Returns:
        nn.Module: The constructed vision projector module.

    Raises:
        ValueError: If the projector type is unknown.
    """
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")
