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
import math
from typing import Any, Dict

import torch
from torch import nn


class SpatialPool(nn.Module):
    def __init__(self, model_args: Dict, vision_tower: nn.Module) -> None:
        """Spatial pooling module for multimodal models.

        Args:
            model_args (Dict): Model arguments containing pooling configuration.
            vision_tower (nn.Module): Vision tower module providing hidden size.
        """
        super().__init__()
        self.mode = model_args.mm_spatial_pool_mode
        self.stride = model_args.mm_spatial_pool_stride
        self.out_channels = getattr(model_args, "mm_spatial_pool_out_channels", vision_tower.hidden_size)

        if self.mode == "average":
            self.pool = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)
        elif self.mode == "max":
            self.pool = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)
        elif self.mode == "conv":
            self.pool = nn.Conv2d(in_channels=vision_tower.hidden_size, out_channels=self.out_channels, kernel_size=self.stride, stride=self.stride)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pool}.")

    def forward(self, x_features: torch.Tensor, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass for processing input image features and tensor x, applying spatial pooling, and returning the output.

        Args:
            x_features (torch.Tensor): The input tensor containing feature maps of shape (B, F, H, W).
            x (torch.Tensor): The input tensor containing image data, shape (B, C, H, W).
            *args (Any): Additional positional arguments (not used in this method).
            **kwargs (Any): Additional keyword arguments (not used in this method).

        Returns:
            torch.Tensor: The processed tensor after applying spatial pooling, reshaped, and transposed.
        """
        # Calculate original width and height based on feature map and image shape
        origin_width = int(math.sqrt(x_features.shape[1] * x.shape[3] // x.shape[2]))
        origin_height = int(origin_width * x.shape[2] // x.shape[3])

        batch_size, _, frame = x_features.shape

        # Reshape and permute the image features to prepare for spatial pooling
        image_features_spatial = x_features.view(batch_size, origin_height, origin_height, frame).permute(0, 3, 1, 2)

        # Apply spatial pooling on the reshaped image features
        image_features_spatial_pool = self.pool(image_features_spatial)

        # Flatten the spatial dimensions and transpose the tensor for further processing
        return image_features_spatial_pool.flatten(2).transpose(1, 2).contiguous()

    @property
    def config(self) -> Dict[str, Any]:
        """Property that returns the configuration of the module.

        Returns:
            Dict[str, Any]: The configuration dictionary containing the type of resampler, stride, pooling mode and output channels.
        """
        return {
            "mm_resampler_type": "spatial_pool",
            "mm_spatial_pool_stride": self.stride,
            "mm_spatial_pool_mode": self.mode,
            "mm_spatial_pool_out_channels": self.out_channels,
        }

    @property
    def hidden_size(self) -> int:
        """Property that returns the hidden size (number of output channels).

        Returns:
            int: The number of output channels after spatial pooling.
        """
        return self.out_channels
