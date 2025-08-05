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
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import requests
import torch
import torch.distributed as dist
import transformers
from PIL import Image
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from llava.constants import IMAGE_TOKEN_INDEX
from .events import LOGGER

__all__ = [
    "convert_expand_to_square", "divide_to_patches", "find_all_linear_names", "load_image", "load_image_from_base64", "get_anyres_image_grid_shape",
    "get_model_name_from_path", "get_peft_state_maybe_zero_3", "get_peft_state_non_lora_maybe_zero_3", "get_mm_adapter_state_maybe_zero_3",
    "maybe_zero_3", "process_anyres_image", "process_images", "rank0_print", "resize_and_pad_image", "select_best_resolution", "split_to_even_chunks",
    "tokenizer_image_token", "unpad_image",
]


def convert_expand_to_square(pil_image: Image.Image, background_color: Union[Tuple[int, int, int], int]) -> Image.Image:
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


def find_all_linear_names(model: transformers.PreTrainedModel) -> List[str]:
    """Find all linear module names in the model.

    Args:
        model (transformers.PreTrainedModel): The model to search for linear modules.

    Returns:
        List[str]: A list of names of linear modules in the model.
    """
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def load_image(image_file) -> Image.Image:
    """Load an image from a local file path or a URL.

    Args:
        image_file (str): Path to the image file or a URL.

    Returns:
        Image.Image: Loaded image in RGB mode.

    Raises:
        IOError: If the image cannot be opened.
        requests.RequestException: If the image URL cannot be fetched.
    """
    try:
        if image_file.startswith(("http://", "https://")):
            response = requests.get(image_file)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_file)
        return image.convert("RGB")
    except Exception as e:
        raise IOError(f"Failed to load image from {image_file}: {e}")


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


def get_peft_state_maybe_zero_3(named_params: Iterable, bias: str) -> Dict[str, torch.Tensor]:
    """Collects LoRA and/or bias parameters from named parameters, handling DeepSpeed Zero3 if needed.

    Args:
        named_params (Iterable): Iterable of (name, parameter) tuples from the model.
        bias (str): Which bias parameters to include. Options:
            - "none": Only LoRA parameters.
            - "all": LoRA and all bias parameters.
            - "lora_only": LoRA parameters and their corresponding bias.

    Returns:
        Dict[str, torch.Tensor]: A dictionary mapping parameter names to (possibly gathered) tensors.
    """
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        bias_name = None
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params: Iterable, require_grad_only: bool = True) -> Dict[str, torch.Tensor]:
    """Collects non-LoRA parameters from named parameters, handling DeepSpeed Zero3 if needed.

    Args:
        named_params (Iterable): Iterable of (name, parameter) tuples from the model.
        require_grad_only (bool): If True, only include parameters that require gradients.

    Returns:
        Dict[str, torch.Tensor]: A dictionary mapping parameter names to (possibly gathered) tensors.
    """
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(
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
                LOGGER.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}.")
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
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
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
            image = convert_expand_to_square(image, tuple(int(x * 255) for x in image_processor.image_mean))
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


def rank0_print(*args) -> None:
    """Prints messages only from the process with rank 0.

    Args:
        *args (Any): Variable length argument list to be printed.
    """
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


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


def unpad_image(x: torch.Tensor, original_size: Tuple[int, int]) -> torch.Tensor:
    """Remove padding from an image tensor to restore its original size.

    Args:
        x (torch.Tensor): The padded image tensor.
        original_size (Tuple[int, int]): The original width and height.

    Returns:
        torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = x.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = x[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = x[:, :, padding:current_width - padding]

    return unpadded_tensor
