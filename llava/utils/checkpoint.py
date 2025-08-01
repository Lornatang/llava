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
from typing import Any, Tuple

import torch
from transformers import AutoTokenizer

from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.models.llm import LlavaLlamaForCausalLM, LlavaQwen2ForCausalLM

__all__ = [
    "load_pretrained",
]


def load_pretrained(
        model: str,
        load_in_8bit: bool = False,
        device: str = "cuda",
        attn_implementation: str = None,
) -> Tuple[Any, Any, Any]:
    """Loads a pretrained model, tokenizer, and image processor.

    Args:
        model (str): Path to the pretrained model.
        load_in_8bit (bool, optional): Whether to load the model in 8-bit mode. Defaults to False.
        device (str, optional): Device to load the model on. Defaults to "cuda".
        attn_implementation (str, optional): Attention implementation type. Defaults to "".

    Returns:
        Tuple[Any, Any, Any]: A tuple containing the tokenizer, model and image processor.
    """
    torch_dtype = torch.float16 if not load_in_8bit else "auto"

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    if "qwen2" in model.lower():
        model = LlavaQwen2ForCausalLM.from_pretrained(
            model,
            low_cpu_mem_usage=True,
            load_in_8bit=load_in_8bit,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            device_map=device,
        )
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model,
            low_cpu_mem_usage=True,
            load_in_8bit=load_in_8bit,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            device_map=device,
        )

    if getattr(model.config, "mm_use_im_patch_token", True):
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if getattr(model.config, "mm_use_im_start_end", False):
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device)
    image_processor = vision_tower.image_processor

    return tokenizer, model, image_processor
