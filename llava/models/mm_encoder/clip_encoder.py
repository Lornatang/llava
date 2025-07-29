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
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    @torch.no_grad()
    def forward(self, images: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Extracts features from input images using the vision tower.

        Args:
            images (Union[torch.Tensor, List[torch.Tensor]]): Input image tensor or list of image tensors.

        Returns:
            torch.Tensor: Extracted image features.
        """
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    def load_model(self, device_map: Any = None) -> None:
        """Loads the CLIP vision tower model and its image processor.

        Args:
            device_map (Any, optional): Device map for model loading. Defaults to None.
        """
        if self.is_loaded:
            print(f"{vision_tower_name} is already loaded, `load_model` called again, skipping.")
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
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
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
    def hidden_size(self) -> int:
        """Returns the hidden size of the vision tower model.

        Returns:
            int: The hidden size.
        """
        return self.config.hidden_size

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
        return (self.config.image_size // self.config.patch_size) ** 2
