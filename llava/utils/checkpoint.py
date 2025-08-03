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
from pathlib import Path
from typing import Any, Tuple

import torch
import transformers
from transformers import AutoTokenizer

from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.models.llm import LlavaLlamaForCausalLM, LlavaQwen2ForCausalLM
from .ops import get_mm_adapter_state_maybe_zero_3

__all__ = [
    "load_pretrained", "safe_save_model_for_hf_trainer",
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


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str) -> None:
    """Collects the state dict and dump to disk.

    Args:
        trainer (transformers.Trainer): The trainer instance.
        output_dir (str): The directory where the model should be saved.
    """
    output_path = Path(output_dir)

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ["mm_projector"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        output_name = output_path.name
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if output_name.startswith("checkpoint-"):
                mm_projector_folder = Path(output_path.parent, "mm_projector")
                mm_projector_folder.mkdir(parents=True, exist_ok=True)
                save_path = Path(mm_projector_folder, f"{output_name}.bin")
                torch.save(weight_to_save, save_path)
            else:
                save_path = Path(output_path, "mm_projector.bin")
                torch.save(weight_to_save, save_path)
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
