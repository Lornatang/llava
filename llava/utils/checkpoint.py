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
from typing import Any, Dict, Optional, Tuple

import torch
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig

from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.models.llm import LlavaLlamaConfig, LlavaLlamaForCausalLM, LlavaQwen2Config, LlavaQwen2ForCausalLM
from .events import LOGGER
from .ops import get_mm_adapter_state_maybe_zero_3, rank0_print

__all__ = [
    "load_pretrained", "safe_save_model_for_hf_trainer",
]


def load_pretrained(
        model_path: str,
        load_8bit: bool = False,
        load_4bit: bool = False,
        device_map: str = "auto",
        torch_dtype: str = "float16",
        attn_implementation: Optional[str] = None,
        customized_config: Optional[Dict] = None,
        overwrite_config: Optional[Dict] = None, **kwargs
) -> Tuple[Any, Any, Any, int]:
    """Loads a pretrained model, tokenizer, and image processor.

    Args:
        model_path (str): Path to the pretrained model.
        load_8bit (bool, optional): Whether to load the model in 8-bit mode. Defaults to ``False``.
        load_4bit (bool, optional): Whether to load the model in 4-bit mode. Defaults to ``False``.
        device_map (str, optional): Device map for model loading. Defaults to ``auto``.
        torch_dtype (str, optional): Data type for the model. Defaults to ``float16``.
        attn_implementation (Optional[str], optional): Attention implementation to use. Defaults to ``None``.
        customized_config (Optional[Dict], optional): Custom configuration for the model. Defaults to ``None``
        overwrite_config (Optional[Dict], optional): Configuration to overwrite the model's config. Defaults to ``None``.

    Returns:
        Tuple[Any, Any, Any, int]: A tuple containing the tokenizer, model, image processor and context length.
    """
    kwargs["device_map"] = device_map

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif torch_dtype == "float16":
        kwargs["torch_dtype"] = torch.float16
    elif torch_dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.float16

    if customized_config is not None:
        kwargs["config"] = customized_config

    if "lora" in model_path.lower():  # TODO: Support LoRA.
        raise "Lora model is not supported in this function."
    else:
        if "qwen2" in model_path.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if "moe" in model_path.lower() or "A14B" in model_path.lower():
                raise "LlavaQwen2MoeForCausalLM is not supported in this function."
                # if overwrite_config is not None:
                #     llava_cfg = LlavaQwen2MoeConfig.from_pretrained(model_path)
                #     LOGGER.info(f"Overwriting config with {overwrite_config}.")
                #     for k, v in overwrite_config.items():
                #         setattr(llava_cfg, k, v)
                #     model = LlavaQwen2MoeForCausalLM.from_pretrained(
                #         model_path,
                #         low_cpu_mem_usage=True,
                #         attn_implementation=attn_implementation,
                #         config=llava_cfg,
                #         **kwargs,
                #     )
                # else:
                #     model = LlavaQwen2MoeForCausalLM.from_pretrained(
                #         model_path,
                #         low_cpu_mem_usage=True,
                #         attn_implementation=attn_implementation,
                #         **kwargs,
                #     )
            else:
                if overwrite_config is not None:
                    llava_cfg = LlavaQwen2Config.from_pretrained(model_path)
                    LOGGER.info(f"Overwriting config with {overwrite_config}.")
                    for k, v in overwrite_config.items():
                        setattr(llava_cfg, k, v)

                    model = LlavaQwen2ForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        attn_implementation=attn_implementation,
                        config=llava_cfg,
                        **kwargs,
                    )
                else:
                    model = LlavaQwen2ForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        attn_implementation=attn_implementation,
                        **kwargs,
                    )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

            if customized_config is None:
                llava_cfg = LlavaLlamaConfig.from_pretrained(model_path)
            else:
                llava_cfg = customized_config

            if overwrite_config is not None:
                LOGGER.info(f"Overwriting config with {overwrite_config}.")
                for k, v in overwrite_config.items():
                    setattr(llava_cfg, k, v)

            model = LlavaLlamaForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                config=llava_cfg,
                **kwargs,
            )

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != "auto":
        vision_tower.to(device="cuda", dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_length = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_length = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_length = model.config.tokenizer_model_max_length
    else:
        context_length = 2048

    return tokenizer, model, image_processor, context_length


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str) -> None:
    """Collects the state dict and dump to disk.

    Args:
        trainer (transformers.Trainer): The trainer instance.
        output_dir (str): The directory where the model should be saved.
    """
    output_path = Path(output_dir)

    if hasattr(trainer.args, "tune_mm_mlp_adapter") and trainer.args.tune_mm_mlp_adapter:
        check_only_save_mm_adapter_tunnable = True
    elif hasattr(trainer.args, "mm_tunable_parts") and (len(trainer.args.mm_tunable_parts.split(",")) == 1 and (
            "mm_mlp_adapter" in trainer.args.mm_tunable_parts or "mm_vision_resampler" in trainer.args.mm_tunable_parts)):
        check_only_save_mm_adapter_tunnable = True
    else:
        check_only_save_mm_adapter_tunnable = False

    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    rank0_print(f"Only save projectors: {check_only_save_mm_adapter_tunnable}")
    if check_only_save_mm_adapter_tunnable:
        # Only save Adapter
        keys_to_match = ["mm_projector", "vision_resampler"]
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
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
