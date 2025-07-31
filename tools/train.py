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
import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.utils.data
import transformers
from PIL import Image
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.engine.trainer import LLaVATrainer
from llava.models.llm.llama import LlavaLlamaForCausalLM
from llava.utils.ops import convert_expand_to_square, tokenizer_image_token
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from transformers import BitsAndBytesConfig

local_rank = None

__all__ = [
    "ModelArguments", "DataArguments", "DataCollatorForSupervisedDataset", "TrainingArguments", "LazySupervisedDataset", "find_all_linear_names",
    "get_peft_state_maybe_zero_3", "maybe_zero_3", "make_supervised_data_module", "safe_save_model_for_hf_trainer",
    "smart_tokenizer_and_embedding_resize", "preprocess", "preprocess_multimodal", "preprocess_plain", "preprocess_vicuna_v1", "preprocess_llama_2",
    "preprocess_deepseek_r1", "preprocess_qwen_2",
]


def _rank0_print(*args) -> None:
    """Prints messages only from the process with rank 0.

    Args:
        *args (Any): Variable length argument list to be printed.
    """
    if local_rank == 0:
        print(*args)


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict[str, Sequence[torch.Tensor]]:
    """Tokenize a list of strings.

    Args:
        strings (Sequence[str]): A list of strings to be tokenized.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

    Returns:
        Dict[str, Sequence[torch.Tensor]]: A dictionary containing the tokenized input IDs and their lengths.
    """
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target: torch.Tensor, tokenized_lens: List[int], speakers: List[str]) -> torch.Tensor:
    """Mask the targets based on tokenized lengths and speakers.

    Args:
        target (torch.Tensor): The target tensor to be masked.
        tokenized_lens (List[int]): A list of tokenized lengths for each round.
        speakers (List[str]): A list of speakers corresponding to each round.

    Returns:
        None: The function modifies the target tensor in place.
    """
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header: str, source: List, get_conversation: bool = True) -> str:
    """Add speaker and start/end signal on each round.

    Args:
        header (str): The header to be added at the beginning of the conversation.
        source (List): A list of sentences, each sentence is a dictionary with "from" and "value" keys.
        get_conversation (bool): Whether to return the full conversation string or just the formatted sentences.

    Returns:
        str: The formatted conversation string with speakers and signals added.
    """
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-13b-v1.5")
    version: Optional[str] = field(default="vicuna_v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of instances into a single batch.

        Args:
            instances (Sequence[Dict]): A sequence of instances, each instance is a dictionary containing "input_ids" and "labels".

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the collated input IDs, labels, attention mask, and optionally images.
        """
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images
        return batch


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


class LazySupervisedDataset(torch.utils.data.Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
            data_path: str,
            tokenizer: transformers.PreTrainedTokenizer,
            data_args: DataArguments,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_path (str): Path to the JSON file containing the dataset.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.
            data_args (DataArguments): The data arguments containing paths and configurations.
        """
        super().__init__()
        _rank0_print("Formatting inputs...Skip in lazy mode.")
        self.tokenizer: transformers.PreTrainedTokenizer = tokenizer
        self.list_data_dict: List[Dict[str, str]] = json.load(open(data_path, "r"))
        self.data_args: DataArguments = data_args

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        """Get an item from the dataset.

        Args:
            i (int): The index of the item to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the input IDs, labels, and optionally images.
        """
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"
        image = None
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(Path(image_folder, image_file)).convert("RGB")
            if self.data_args.image_aspect_ratio == "pad":
                image = convert_expand_to_square(image, tuple(int(x * 255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            else:
                image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(sources, self.tokenizer, ("image" in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # Image exists in the data.
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # The image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])
        return data_dict

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.list_data_dict)

    @property
    def lengths(self) -> List[int]:
        """Return the lengths of each sample in the dataset.

        Returns:
            List[int]: A list of lengths for each sample, where each length is the number of tokens in the conversations.
        """
        length_list = []
        for sample in self.list_data_dict:
            image_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + image_tokens)
        return length_list

    @property
    def modality_lengths(self) -> List[int]:
        """Return the modality lengths of each sample in the dataset.

        Returns:
            List[int]: A list of lengths for each sample, where each length is the number of tokens in the conversations.
            The length is negative if the sample contains an image, otherwise it is positive.
        """
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list


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


def get_mm_adapter_state_maybe_zero_3(named_params: Iterable, keys_to_match: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
    """Collects multimodal adapter parameters from named parameters, handling DeepSpeed Zero3 if needed.

    Args:
        named_params (Iterable): Iterable of (name, parameter) tuples from the model.
        keys_to_match (Optional[List[str]]): List of substrings to match in parameter names. If None, all parameters are returned.

    Returns:
        Dict[str, torch.Tensor]: A dictionary mapping parameter names to (possibly gathered) tensors.
    """
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def maybe_zero_3(param: torch.nn.Parameter, ignore_status: bool = False) -> torch.Tensor:
    """Handles DeepSpeed Zero3 parameters.

    Args:
        param (torch.nn.Parameter): The parameter to process.
        ignore_status (bool): If True, ignores the ZeroParamStatus check.

    Returns:
        torch.Tensor: The processed parameter tensor, detached and moved to CPU.
    """
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments) -> Dict[str, torch.utils.data.Dataset]:
    """Create a dataset and collator for supervised fine-tuning.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.
        data_args (DataArguments): The data arguments containing paths and configurations.

    Returns:
        Dict[str, torch.utils.data.Dataset]: A dictionary containing the training dataset, evaluation dataset (if any), and data collator.
    """
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )


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


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
) -> None:
    """Resize tokenizer and embedding.

    Args:
        special_tokens_dict (Dict): A dictionary of special tokens to be added to the tokenizer.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to resize.
        model (transformers.PreTrainedModel): The model whose embeddings will be resized.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def preprocess(
        sources: Dict[str, List[Dict[str, str]]],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
) -> Dict[str, torch.Tensor]:
    """Preprocess conversations for supervised fine-tuning.

    Args:
        sources (Dict[str, List[Dict[str, str]]]): A list of conversations, each conversation is a list of sentences.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.
        has_image (bool): Whether the conversations contain images.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the tokenized input IDs and labels.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.VICUNA_V1:
        return preprocess_vicuna_v1(sources, tokenizer, has_image)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image)
    # TODO: implemenmts DeepSeek process function.
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.DEEPSEEK_R1:
        pass
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.QWEN_2:
        return preprocess_qwen_2(sources, tokenizer, has_image)

    # add end signal and concatenate together
    conversations = []
    header = None
    for source in sources:
        header = f"{conversation_lib.default_conversation.system_message}\n\n"
        conversation = _add_speaker_and_signal(header, source, version)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_multimodal(
        sources: Dict[str, List[Dict[str, str]]],
        data_args: DataArguments,
) -> Dict[str, torch.Tensor]:
    """Preprocess conversations for multimodal fine-tuning.

    Args:
        sources (Dict[str, List[Dict[str, str]]]): A list of conversations, each conversation is a list of sentences.
        data_args (DataArguments): The data arguments containing multimodal configurations.

    Returns:
        Dict[str, List[Dict[str, str]]]: The preprocessed conversations with image tokens replaced.
    """
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_plain(
        sources: Dict[str, List[Dict[str, str]]],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict[str, List[Dict[str, str]]]:
    """Preprocess conversations in plain style.

    Args:
        sources (Dict[str, List[Dict[str, str]]]): A list of conversations, each conversation is a list of sentences.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

    Returns:
        Dict[str, List[Dict[str, str]]]: A dictionary containing the tokenized input IDs and labels.
    """
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_vicuna_v1(
        sources: Dict[str, List[Dict[str, str]]],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
) -> Dict[str, List[Dict[str, str]]]:
    """Preprocess conversations in Vicuna v1 style.

    Args:
        sources (Dict[str, List[Dict[str, str]]]): A list of conversations, each conversation is a list of sentences.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.
        has_image (bool): Whether the conversations contain images.

    Returns:
        Dict[str, List[Dict[str, str]]]: A dictionary containing the tokenized input IDs and labels.
    """
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.VICUNA_V1

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                round_len -= 1
                instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama_2(
        sources: Dict[str, List[Dict[str, str]]],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
) -> Dict[str, List[Dict[str, str]]]:
    """Preprocess conversations in Llama 2 style.

    Args:
        sources (Dict[str, List[Dict[str, str]]]): A list of conversations, each conversation is a list of sentences.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.
        has_image (bool): Whether the conversations contain images.

    Returns:
        Dict[str, List[Dict[str, str]]]: A dictionary containing the tokenized input IDs and labels.
    """
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_deepseek_r1(
        sources: Dict[str, List[Dict[str, str]]],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
) -> Dict[str, List[Dict[str, str]]]:
    """Preprocess conversations in DeepSeek-R1 style.

    Args:
        sources (Dict[str, List[Dict[str, str]]]): A list of conversations, each conversation is a list of sentences.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.
        has_image (bool): Whether the conversations contain images.

    Returns:
        Dict[str, List[Dict[str, str]]]: A dictionary containing the tokenized input IDs and labels.
    """
    pass


def preprocess_qwen_2(
        sources: Dict[str, List[Dict[str, str]]],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
) -> Dict[str, List[Dict[str, str]]]:
    """Preprocess conversations in Qwen2 style.

    Args:
        sources (Dict[str, List[Dict[str, str]]]): A list of conversations, each conversation is a list of sentences.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.
        has_image (bool): Whether the conversations contain images.

    Returns:
        Dict[str, List[Dict[str, str]]]: A dictionary containing the tokenized input IDs and labels.
    """
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.QWEN_2

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        rounds_len = len(rounds)
        cur_len = 0
        # target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_ids = tokenizer_image_token(rou, tokenizer)
                instruction_ids = tokenizer_image_token(parts[0], tokenizer)
                equal_parts = [x == y for x, y in zip(round_ids, instruction_ids)]

                instruction_len = equal_parts.index(False) if False in equal_parts else len(equal_parts)
                round_len = len(round_ids)
            else:
                round_ids = tokenizer(rou).input_ids
                instruction_ids = tokenizer(parts[0]).input_ids
                equal_parts = [x == y for x, y in zip(round_ids, instruction_ids)]

                instruction_len = equal_parts.index(False) if False in equal_parts else len(equal_parts)
                round_len = len(round_ids)

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len + rounds_len - 2:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def train(attn_implementation: str = None) -> None:
    """Main training function to set up the model, tokenizer, and training arguments.

    Args:
        attn_implementation (str): The attention implementation to use, e.g., "flash_attention_2".
    """
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type  # {"fp4", "nf4"}
                )
            )
        )

    if model_args.vision_tower is not None:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            local_files_only=True,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            local_files_only=True,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        _rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
    else:  # use qwen
        tokenizer.legacy = False
    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = LLaVATrainer(model=model, processing_class=tokenizer, args=training_args, **data_module)

    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, Path(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
