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

__all__ = [
    "PoolerProjector",
]


class PoolerProjector(nn.Module):
    """Pooler projector module for vision feature projection."""

    def __init__(self, config: Dict, vision_cfg: Dict) -> None:
        """Initializes the PoolerProjector module.

        Args:
            config (Dict): Configuration dictionary containing model parameters.
            vision_cfg (Dict): Configuration dictionary for vision model parameters.
        """
        super().__init__()
        self._config = config
        self.hw = vision_cfg.image_size // vision_cfg.patch_size
        self.conv_pool = nn.Conv2d(config.mm_hidden_size, config.hidden_size, kernel_size=2, stride=2)
        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Projects the input tensor using convolutional pooling and a linear layer.

        Args:
            x (torch.Tensor): Input tensor.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The projected tensor.
        """
        height = width = self.hw
        assert height * width == x.shape[1]
        x = x.view(x.shape[0], height, width, -1).permute(0, 3, 1, 2)
        x = self.conv_pool(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

    @property
    def config(self) -> Dict[str, str]:
        """Returns the configuration for the pooler projector.

        Returns:
            Dict[str, str]: Configuration dictionary.
        """
        return {"mm_projector_type": "pooler"}
