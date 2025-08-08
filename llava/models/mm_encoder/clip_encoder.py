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
from typing import Any, List, Union

import torch
from torch import nn
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

from llava.utils.events import LOGGER

__all__ = [
    "CLIPVisionTower",
]


class CLIPVisionTower(nn.Module):
    """Vision tower module based on CLIP for extracting image features."""

    def __init__(
            self,
            vision_tower: str,
            args: nn.Module,
            delay_load: bool = False,
    ) -> None:
        """ Initializes the CLIPVisionTower.

        Args:
            vision_tower (str): The name or path of the vision tower model.
            args (nn.Module): Arguments containing configuration for the vision tower.
            delay_load (bool, optional): If True, the model will not be loaded immediately. Defaults to ``False``.
        """
        super().__init__()
        self.is_loaded: bool = False

        self.vision_tower_name: str = vision_tower
        self.select_layer: int = args.mm_vision_select_layer
        self.select_feature: str = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            LOGGER.info(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            LOGGER.info(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
            LOGGER.info(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    @torch.no_grad()
    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Extracts features from input images using the vision tower.

        Args:
            x (Union[torch.Tensor, List[torch.Tensor]]): Input image tensor or list of image tensors.

        Returns:
            torch.Tensor: Extracted image features.
        """
        if type(x) is list:
            x_features = []
            for image in x:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                x_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(x.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            x_features = self.feature_select(image_forward_outs).to(x.dtype)

        return x_features

    def load_model(self, device_map: Any = None) -> None:
        """Loads the CLIP vision tower model and its image processor.

        Args:
            device_map (Any, optional): Device map for model loading. Defaults to None.
        """
        if self.is_loaded:
            LOGGER.info(f"{self.vision_tower_name} is already loaded, `load_model` called again, skipping.")
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs: Any) -> torch.Tensor:
        """
        Selects features from the model output according to the configured feature type.

        Args:
            image_forward_outs (Any): The output object from the vision tower forward pass,
                expected to have a `hidden_states` attribute.

        Returns:
            torch.Tensor: The selected image features tensor.

        Raises:
            ValueError: If `self.select_feature` is not a supported type.
        """
        select_feature_type = self.select_feature

        if self.select_feature in ["slicefour_patch", "slicefour_cls_patch"]:
            select_every_k_layer = len(image_forward_outs.hidden_states) // 4
            image_features = torch.cat(
                [
                    image_forward_outs.hidden_states[i]
                    for i in range(select_every_k_layer + self.select_layer, len(image_forward_outs.hidden_states), select_every_k_layer)
                ],
                dim=-1)
            select_feature_type = select_feature_type.replace("slicefour_", "")
        elif self.select_feature in ["slice_m25811_f6_patch", "slice_m25811_f6_cls_patch"]:
            select_layers = [-2, -5, -8, -11, 6]
            image_features = torch.cat([image_forward_outs.hidden_states[i] for i in select_layers], dim=-1)
            select_feature_type = select_feature_type.replace("slice_m25811_f6_", "")
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]

        if select_feature_type == "patch":
            image_features = image_features[:, 1:]
        elif select_feature_type == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {select_feature_type}")
        return image_features

    @property
    def dummy_feature(self) -> torch.Tensor:
        """Returns a dummy feature tensor for the vision tower.

        Returns:
            torch.Tensor: A zero tensor with shape (1, hidden_size).
        """
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self) -> torch.dtype:
        """Returns the data type of the vision tower model.

        Returns:
            torch.dtype: The data type.
        """
        return self.vision_tower.dtype

    @property
    def device(self) -> torch.device:
        """Returns the device of the vision tower model.

        Returns:
            torch.device: The device.
        """
        return self.vision_tower.device

    @property
    def config(self) -> Any:
        """Returns the configuration of the vision tower model.

        Returns:
            Any: The model configuration object.
        """
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def image_size(self):
        return self.config.image_size

    @property
    def hidden_size(self) -> int:
        """Returns the hidden size of the vision tower model.

        Returns:
            int: The hidden size.
        """
        _hidden_size = self.config.hidden_size
        if "slicefour" in self.select_feature:
            _hidden_size *= 4
        if "slice_m25811_f6" in self.select_feature:
            _hidden_size *= 5
        return _hidden_size

    @property
    def num_patches_per_side(self) -> int:
        """Returns the number of patches per image side.

        Returns:
            int: Number of patches per side.
        """
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self) -> int:
        """Returns the total number of patches in the image.

        Returns:
            int: Total number of patches.
        """
        _num_patches = (self.config.image_size // self.config.patch_size) ** 2
        if "cls_patch" in self.select_feature:
            _num_patches += 1
        return _num_patches
