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
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..arch import LlavaMetaModel, LlavaMetaForCausalLM

__all__ = [
    "LlavaQwen2Config", "LlavaQwen2Model", "LlavaQwen2ForCausalLM",
]


class LlavaQwen2Config(Qwen2Config):
    """Configuration class for llava_qwen2 model."""
    model_type: str = "llava_qwen2"


class LlavaQwen2Model(LlavaMetaModel, Qwen2Model):
    """LlavaQwen2Model model class."""
    config_class: LlavaQwen2Config = LlavaQwen2Config

    def __init__(self, config: Qwen2Config) -> None:
        """Initialize the LlavaQwen2Model model.

        Args:
            config (Qwen2Config): Configuration for the LlavaQwen2Model model.
        """
        super().__init__(
            config=config,
        )


class LlavaQwen2ForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    """LlavaQwen2ForCausalLM model for causal language modeling."""
    config_class: LlavaQwen2Config = LlavaQwen2Config

    def __init__(self, config: Qwen2Config) -> None:
        """Initialize the LlavaQwen2ForCausalLM model for causal language modeling.

        Args:
            config (Qwen2Config): Configuration for the LlavaQwen2ForCausalLM model.
        """
        super().__init__(
            config=config,
        )
        config.model_type = "llava_qwen2"
        config.rope_scaling = None

        self.model: LlavaQwen2Model = LlavaQwen2Model(config)
        self.lm_head: nn.Linear = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing.
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            image_sizes: Optional[List[Tuple[int, int]]] = None,
            return_dict: Optional[bool] = None,
            modalities: Optional[List[str]] = None,
            dpo_forward: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Forward pass for the Llava Llama model.

        Args:
            input_ids (torch.LongTensor, optional): Input token IDs. Defaults to ``None``.
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to ``None``.
            position_ids (Optional[torch.LongTensor], optional): Position IDs. Defaults to ``None``.
            past_key_values (Optional[List[torch.FloatTensor]], optional): Past key values for caching. Defaults to ``None``.
            inputs_embeds (Optional[torch.FloatTensor], optional): Input embeddings. Defaults to ``None``.
            cache_position (Optional[torch.LongTensor], optional): Cache position for the model. Defaults to ``None``.
            use_cache (Optional[bool], optional): Whether to use cache. Defaults to ``None``.
            labels (Optional[torch.LongTensor], optional): Labels for the model. Defaults to ``None``.
            output_attentions (Optional[bool], optional): Whether to output attentions. Defaults to ``None``.
            output_hidden_states (Optional[bool], optional): Whether to output hidden states. Defaults to ``None``.
            images (Optional[torch.FloatTensor], optional): Input images for multimodal processing. Defaults to ``None``.
            image_sizes (Optional[List[Tuple[int, int]]], optional): Sizes of the input images. Defaults to ``None``.
            return_dict (Optional[bool], optional): Whether to return a dictionary of outputs. Defaults to ``None``.
            modalities (Optional[List[str]], optional): List of modalities to process. Defaults to ``None``.
            dpo_forward (Optional[bool], optional): Whether to perform DPO forward pass. Defaults to ``False``.

        Returns:
            Union[Tuple, CausalLMOutputWithPast]: Model outputs, either as a tuple or a CausalLMOutputWithPast object.
        """
        if modalities is None:
            modalities = ["image"]

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                modalities,
            )

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels
        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    def get_model(self) -> LlavaQwen2Model:
        """Get the underlying Llava Llama model."""
        return self.model

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            **kwargs,
    ) -> Dict[str, Union[torch.LongTensor, torch.FloatTensor]]:
        """Prepares inputs for generation, supporting multimodal information.

        Args:
            input_ids (torch.LongTensor): Input token IDs.
            past_key_values (Optional[List[torch.FloatTensor]], optional): Cached past key values. Defaults to ``None``.
            inputs_embeds (Optional[torch.FloatTensor], optional): Input embeddings. Defaults to ``None``.
            **kwargs: Additional keyword arguments may include 'images' (image features) and 'image_sizes' (image sizes).

        Returns:
            Dict[str, Union[torch.LongTensor, torch.FloatTensor]]: Dictionary containing all required inputs for generation, including multimodal information if provided.
        """
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)

        if images is not None:
            inputs["images"] = images

        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            images: Optional[torch.Tensor] = None,
            image_sizes: Optional[List[Tuple[int, int]]] = None,
            modalities: Optional[List[str]] = None,
            **kwargs,
    ) -> GenerateOutput:
        """Generate sequences from the Llava model (no gradient tracking).

        Args:
            inputs (Optional[torch.Tensor], optional): Input token IDs (or `None` if passing `inputs_embeds` via kwargs). Defaults to ``None``.
            images (Optional[torch.Tensor], optional): Batch of input images, each as a tensor. Defaults to ``None``.
            image_sizes (Optional[List[Tuple[int, int]]], optional): List of (width, height) for each image in the batch. Defaults to ``None``.
            modalities (Optional[List[str]], optional): Modalities to process (e.g. ["image", "text"]). Defaults to ``None``.
            **kwargs: Other generation args or any parameters accepted by `transformers.generate`. Defaults to ``None``.

        Returns:
            GenerateOutput: Generation results (with `sequences`, `scores`, etc.), or a raw tensor of token IDs if `return_dict=False`.
        """
        if modalities is None:
            modalities = ["image"]

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                past_key_values=None,
                labels=None,
                images=images,
                image_sizes=image_sizes,
                modalities=modalities,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


AutoConfig.register("llava_qwen2", LlavaQwen2Config)
AutoModelForCausalLM.register(LlavaQwen2Config, LlavaQwen2ForCausalLM)
