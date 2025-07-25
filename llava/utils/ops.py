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
import ast
import base64
import math
from io import BytesIO
from typing import Any, List, Tuple, Union, Optional

import torch
from PIL import Image
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from llava.constants import IMAGE_TOKEN_INDEX

__all__ = [
    "divide_to_patches", "expand2square", "load_image_from_base64", "get_anyres_image_grid_shape", "get_length_grouped_indices",
    "get_model_name_from_path", "get_multi_modality_length_grouped_indices", "get_multi_modality_adapter_state_maybe_zero_3", "maybe_zero_3",
    "process_anyres_image", "process_images", "resize_and_pad_image", "select_best_resolution", "tokenizer_image_token"
]


def divide_to_patches(image: Image.Image, patch_size: int) -> List[Image.Image]:
    """Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        List[PIL.Image.Image]: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def expand2square(pil_image: Image.Image, background_color: Union[Tuple[int, int, int], int]) -> Image.Image:
    """Expands a PIL image to a square by padding with a background color.

    Args:
        pil_image (PIL.Image.Image): The input image.
        background_color (Union[Tuple[int, int, int], int]): The background color for padding.

    Returns:
        PIL.Image.Image: The squared image.
    """
    width, height = pil_image.size
    if width == height:
        return pil_image
    elif width > height:
        result = Image.new(pil_image.mode, (width, width), background_color)
        result.paste(pil_image, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_image.mode, (height, height), background_color)
        result.paste(pil_image, ((height - width) // 2, 0))
        return result


def load_image_from_base64(image: str) -> Image.Image:
    """Loads an image from a base64-encoded string.

    Args:
        image (str): Base64-encoded image string.

    Returns:
        PIL.Image.Image: The loaded image.
    """
    return Image.open(BytesIO(base64.b64decode(image)))


def get_anyres_image_grid_shape(
        image_size: Tuple[int, int],
        grid_pinpoints: Union[str, List[Tuple[int, int]]],
        patch_size: int,
) -> Tuple[int, int]:
    """Calculate the shape of the image patch grid after preprocessing for images of any resolution.

    Args:
        image_size (Tuple[int, int]): The size of the input image (width, height).
        grid_pinpoints (Union[str, List[Tuple[int, int]]]): List or string of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        Tuple[int, int]: The shape of the image patch grid (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


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


def get_model_name_from_path(model_path: str) -> str:
    """Extracts the model name from a model path.

    Args:
        model_path (str): The path to the model.

    Returns:
        str: The extracted model name.
    """
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


def get_multi_modality_length_grouped_indices(
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
        generator (Optional[torch.Generator], optional): Random generator. Defaults to None.

    Returns:
        Union[List[List[Any]], list[int]]: Grouped and shuffled indices.
    """
    assert all(l != 0 for l in lengths), "Should not have zero length."

    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)

    # multi modality sample.
    multi_modality_indices, multi_modality_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    # language sample.
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    multi_modality_shuffle = [
        multi_modality_indices[i] for i in get_length_grouped_indices(multi_modality_lengths, batch_size, world_size, generator=None)
    ]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    multi_modality_mega_batches = [multi_modality_shuffle[i: i + megabatch_size] for i in range(0, len(multi_modality_shuffle), megabatch_size)]
    lang_mega_batches = [lang_shuffle[i: i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_multi_modality = multi_modality_mega_batches[-1]
    last_lang = lang_mega_batches[-1]
    additional_batch = last_multi_modality + last_lang
    mega_batches = multi_modality_mega_batches[:-1] + lang_mega_batches[:-1]
    megabatch_indices = torch.randperm(len(mega_batches), generator=generator)
    mega_batches = [mega_batches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        mega_batches.append(sorted(additional_batch))

    return [i for megabatch in mega_batches for i in megabatch]


def get_multi_modality_adapter_state_maybe_zero_3(
        named_params: List[tuple],
        keys_to_match: List[str]
) -> dict:
    """Extracts and gathers adapter parameters matching specified keys, handling DeepSpeed Zero-3.

    Args:
        named_params (List[tuple]): List of (name, parameter) tuples.
        keys_to_match (List[str]): List of key substrings to match parameter names.

    Returns:
        dict: Dictionary of gathered parameter tensors.
    """
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def maybe_zero_3(param: torch.nn.Parameter, ignore_status: bool = False, name: Optional[str] = None) -> torch.Tensor:
    """Safely gathers and clones a parameter, handling DeepSpeed Zero-3 status.

    Args:
        param (torch.nn.Parameter): The parameter to gather and clone.
        ignore_status (bool, optional): Whether to ignore NOT_AVAILABLE status. Defaults to False.
        name (Optional[str], optional): Parameter name for logging. Defaults to None.

    Returns:
        torch.Tensor: The gathered and cloned parameter tensor.
    """
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def process_anyres_image(
        image: Image.Image,
        processor: Any,
        grid_pinpoints: Union[str, List[Tuple[int, int]]],
) -> torch.Tensor:
    """Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor (Any): The image processor object.
        grid_pinpoints (Union[str, List[Tuple[int, int]]]): List or string of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)

    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)
    patches = divide_to_patches(image_padded, processor.crop_size["height"])
    image_original_resize = image.resize((processor.size["shortest_edge"], processor.size["shortest_edge"]))
    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0]
                     for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def process_images(
        images: List[Image.Image],
        image_processor: Any,
        model_cfg: Any,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Processes a list of images according to the model configuration.

    Args:
        images (List[PIL.Image.Image]): List of input images.
        image_processor (Any): The image processor object.
        model_cfg (Any): Model configuration object.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: Processed images as a tensor or list of tensors.
    """
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            new_images.append(image)
    elif image_aspect_ratio == "anyres":
        for image in images:
            image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    else:
        return image_processor(images, return_tensors="pt")["pixel_values"]

    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def resize_and_pad_image(
        image: Image.Image,
        target_resolution: Tuple[int, int],
) -> Image.Image:
    """Resize and pad an image to a target resolution while maintaining an aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (Tuple[int, int]): The target resolution (width, height).

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def select_best_resolution(
        original_size: Tuple[int, int],
        possible_resolutions: List[Tuple[int, int]],
) -> Tuple[int, int]:
    """Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (Tuple[int, int]): The original size of the image (width, height).
        possible_resolutions (List[Tuple[int, int]]): List of possible resolutions.

    Returns:
        Tuple[int, int]: The best fit resolution (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if (
                effective_resolution > max_effective_resolution or
                (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution)
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def split_to_even_chunks(
        indices: List[int],
        lengths: List[int],
        num_chunks: int
) -> List[List[int]]:
    """Splits indices into even chunks based on their lengths.

    Args:
        indices (List[int]): List of indices to split.
        lengths (List[int]): List of lengths corresponding to indices.
        num_chunks (int): Number of chunks to split into.

    Returns:
        List[List[int]]: List of index chunks.
    """
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def tokenizer_image_token(
        prompt: str,
        tokenizer: Any,
        image_token_index: int = IMAGE_TOKEN_INDEX,
        return_tensors: Optional[str] = None,
) -> Union[List[int], torch.Tensor]:
    """Tokenizes a prompt containing image tokens.

    Args:
        prompt (str): The input prompt string.
        tokenizer (Any): The tokenizer object.
        image_token_index (int, optional): The index for image tokens. Defaults to IMAGE_TOKEN_INDEX.
        return_tensors (Optional[str], optional): The type of tensor to return. Defaults to None.

    Returns:
        Union[List[int], torch.Tensor]: Tokenized prompt as a list or tensor.
    """
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(inputs, sep):
        return [ele for sublist in zip(inputs, [sep] * len(inputs)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids
