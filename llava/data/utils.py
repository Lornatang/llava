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
from typing import Any, List, Optional, Union

import torch
from transformers.trainer_pt_utils import get_length_grouped_indices as get_length_grouped_indices_hf

from llava.utils.ops import split_to_even_chunks

__all__ = [
    "get_length_grouped_indices", "get_length_grouped_indices_auto_single", "get_mm_length_grouped_indices", "get_mm_length_grouped_indices_auto",
    "get_variable_length_grouped_indices",
]


def get_length_grouped_indices(
        lengths: List[int],
        batch_size: int,
        world_size: int,
        generator: Optional[torch.Generator] = None,
) -> List[int]:
    """Groups indices by length for batching.

    Args:
        lengths (List[int]): List of sample lengths.
        batch_size (int): Batch size.
        world_size (int): Number of distributed workers.
        generator (Optional[torch.Generator], optional): Random generator. Defaults to ``None``.

    Returns:
        List[int]: Grouped and shuffled indices.
    """
    # Shuffle indices based on lengths.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    # Split indices into mega_batches.
    mega_batches = [indices[i: i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    # Sort each megabatch by length in descending order.
    mega_batches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in mega_batches]
    # If the last megabatch is smaller than the megabatch size, add it to the previous one.
    mega_batches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in mega_batches]

    return [i for megabatch in mega_batches for batch in megabatch for i in batch]


def get_length_grouped_indices_auto_single(
        lengths: List[int],
        batch_size: int,
        world_size: int,
        generator: Optional[torch.Generator] = None,
) -> List[int]:
    """Groups indices by lengths for single modality batching.

    Args:
        lengths (List[int]): List of sample lengths (positive for multimodal, negative for language).
        batch_size (int): Batch size.
        world_size (int): Number of distributed workers.
        generator (Optional[torch.Generator], optional): Random generator. Defaults to ``None``.

    Returns:
          List[int]: Grouped and shuffled indices for single modality.
    """
    # Shuffle indices based on lengths.
    indices = get_length_grouped_indices_hf(lengths, batch_size * world_size, generator=generator)
    megabatch_size = world_size * batch_size
    # Split indices into mega_batches.
    megabatches = [indices[i: i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    # Sort each megabatch by length in descending order.
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    # If the last megabatch is smaller than the megabatch size, add it to the previous one.
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]
    # Shuffle the megabatches.
    batch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in batch_indices]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_mm_length_grouped_indices(
        lengths: List[int],
        batch_size: int,
        world_size: int,
        generator: Optional[torch.Generator] = None
) -> Union[List[List[Any]], list[int]]:
    """Groups indices by modality and length for batching.

    Args:
        lengths (List[int]): List of sample lengths (positive for multimodal, negative for language).
        batch_size (int): Batch size.
        world_size (int): Number of distributed workers.
        generator (Optional[torch.Generator], optional): Random generator. Defaults to ``None``.

    Returns:
        Union[List[List[Any]], list[int]]: Grouped and shuffled indices.
    """
    assert all(l != 0 for l in lengths), "Should not have zero length."

    # mm sample lengths are positive, lang sample lengths are negative.
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)

    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_mega_batches = [mm_shuffle[i: i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_mega_batches = [lang_shuffle[i: i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_mega_batches[-1]
    last_lang = lang_mega_batches[-1]
    additional_batch = last_mm + last_lang
    mega_batches = mm_mega_batches[:-1] + lang_mega_batches[:-1]
    megabatch_indices = torch.randperm(len(mega_batches), generator=generator)
    mega_batches = [mega_batches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        mega_batches.append(sorted(additional_batch))

    return [i for megabatch in mega_batches for i in megabatch]


def get_mm_length_grouped_indices_auto(
        lengths: List[int],
        batch_size: int,
        world_size: int,
        generator: Optional[torch.Generator] = None
) -> Union[List[List[Any]], list[int]]:
    """Groups indices by modality and length for batching.

    Args:
        lengths (List[int]): List of sample lengths (positive for multimodal, negative for language).
        batch_size (int): Batch size.
        world_size (int): Number of distributed workers.
        generator (Optional[torch.Generator], optional): Random generator. Defaults to ``None``.

    Returns:
        Union[List[List[Any]], list[int]]: Grouped and shuffled indices.
    """
    assert all(l != 0 for l in lengths), "Should not have zero length."

    # mm sample lengths are positive, lang sample lengths are negative.
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        return get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=generator)

    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices_auto_single(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices_auto_single(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i: i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i: i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # Hard code to avoid last batch mixed with different modalities.
    if len(mm_megabatches[-1]) > 0:
        megabatches.append(mm_megabatches[-1])
    if len(lang_megabatches[-1]) > 0:
        megabatches.append(lang_megabatches[-1])

    return [i for megabatch in megabatches for i in megabatch]


def get_variable_length_grouped_indices(
        lengths: List[int],
        batch_size: int,
        world_size: int,
        megabatch_mult: int = 8,
        generator: Optional[torch.Generator] = None,
) -> List[List[Any]]:
    """Groups indices by variable lengths for batching.

    Args:
        lengths (List[int]): List of sample lengths (positive for multimodal, negative for language).
        batch_size (int): Batch size.
        world_size (int): Number of distributed workers.
        megabatch_mult (int, optional): Multiplier for megabatch size. Defaults to 8.
        generator (Optional[torch.Generator], optional): Random generator. Defaults to ``None``.

    Returns:
        List[List[Any]]: Grouped and shuffled indices.
    """
    indices = torch.randperm(len(lengths), generator=generator)
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    megabatch_size = world_size * batch_size * megabatch_mult
    megabatches = [sorted_indices[i: i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: indices[i], reverse=True) for megabatch in megabatches]
    shuffled_indices = [i for megabatch in megabatches for i in megabatch]
    world_batch_size = world_size * batch_size
    batches = [shuffled_indices[i: i + world_batch_size] for i in range(0, len(lengths), world_batch_size)]
    batch_indices = torch.randperm(len(batches), generator=generator)
    batches = [batches[i] for i in batch_indices]

    return [i for batch in batches for i in batch]
