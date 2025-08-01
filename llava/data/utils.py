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
from typing import Any, Iterator, List, Optional

import bitsandbytes
import torch
from torch.utils.data import Sampler

from llava.utils.ops import get_length_grouped_indices, get_mm_length_grouped_indices

__all__ = [
    "LengthGroupedSampler",
]


class LengthGroupedSampler(Sampler):
    """Sampler that groups samples by length or modality."""

    def __init__(
            self,
            batch_size: int,
            world_size: int,
            lengths: Optional[List[int]] = None,
            generator: Optional[torch.Generator] = None,
            group_by_modality: bool = False,
    ) -> None:
        """Initializes a sampler that groups samples by length or modality.

        Args:
            batch_size (int): Batch size.
            world_size (int): Number of distributed workers.
            lengths (Optional[List[int]]): List of sample lengths.
            generator (Optional[torch.Generator], optional): Random generator. Defaults to ``None``.
            group_by_modality (bool, optional): Whether to group by modality. Defaults to ``False``.
        """
        super().__init__()
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size: int = batch_size
        self.world_size: int = world_size
        self.lengths: Optional[List[int]] = lengths
        self.generator: Optional[torch.Generator] = generator
        self.group_by_modality: bool = group_by_modality

    def __len__(self) -> int:
        """Returns the number of samples.

        Returns:
            int: Number of samples.
        """
        return len(self.lengths)

    def __iter__(self) -> Iterator[List[Any]]:
        """Yields indices for sampling.

        Returns:
            Iterator[List[Any]]: Iterator over sample indices.
        """
        if self.group_by_modality:
            indices = get_mm_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)
