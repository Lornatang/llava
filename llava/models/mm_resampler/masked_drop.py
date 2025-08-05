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
import random
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import nn

__all__ = [
    "MaskedDrop",
]


class MaskedDrop(nn.Module):
    """MaskedDrop module for image feature resampling."""

    def __init__(self, model_args: Any) -> None:
        """Initializes the MaskedDrop module.

        Args:
            model_args (Any): Arguments containing mask drop configuration.
        """
        super().__init__()
        self.mode = model_args.mm_mask_drop_mode
        self.skip_percentage = model_args.mm_mask_drop_skip_percentage
        self.ratio = model_args.mm_mask_drop_ratio
        self.ratio_upper = model_args.mm_mask_drop_ratio_upper
        self.ratio_lower = model_args.mm_mask_drop_ratio_lower

    def forward(self, x: Union[List[torch.Tensor], torch.Tensor], *args: Any, **kwargs: Any) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Applies masked drop to the input image features during training.

        Args:
            x (Union[List[torch.Tensor], torch.Tensor]): Input image features.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Union[List[torch.Tensor], torch.Tensor]: Masked image features.
        """
        if not self.training:
            return x

        if self.skip_percentage > random.random():
            return x

        masked_features = []

        for image_feature in x:
            num_tokens = image_feature.shape[0]
            if self.mode == "fixed":
                num_keep = int(num_tokens * self.ratio)
                masked_features.append(self.random_masking(image_feature.unsqueeze(0), num_keep)[0][0])
            elif self.mode == "range":
                num_keep = int(num_tokens * random.uniform(self.ratio_lower, self.ratio_upper))
                masked_features.append(self.random_masking(image_feature.unsqueeze(0), num_keep)[0])
            elif self.mode == "cls_only":
                masked_features.append(image_feature[0:1])
            else:
                raise ValueError(f"Unexpected masked drop mode: {self.mode}")

        if self.mode not in ["range"] and (type(x) is not list or self.mode in ["cls_only"]):
            masked_features = torch.stack(masked_features, dim=0)

        return masked_features

    @property
    def config(self) -> Dict[str, Any]:
        """Returns the configuration of the MaskedDrop module.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        return {
            "mm_resampler_type": "masked_drop",
            "mm_mask_drop_mode": self.mode,
            "mm_mask_drop_skip_percentage": self.skip_percentage,
            "mm_mask_drop_ratio": self.ratio,
            "mm_mask_drop_ratio_upper": self.ratio_upper,
            "mm_mask_drop_ratio_lower": self.ratio_lower,
        }

    def random_masking(self, x: torch.Tensor, len_keep: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly masks tokens in the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (N, L, D).
            len_keep (int): Number of tokens to keep.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Masked tensor of shape (N, len_keep, D).
                - Binary mask of shape (N, L).
                - Restore indices of shape (N, L).
        """
        batch_size, length, dim = x.shape

        noise = torch.rand(batch_size, length, device=x.device)  # noise in [0, 1].

        # sort noise for each sample.
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset.
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is removed.
        mask = torch.ones([batch_size, length], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask.
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
