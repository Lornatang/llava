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
import random
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import math
import torch
import torch.utils.data
import transformers
import yaml
from PIL import Image, ImageFile
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer

from llava import conversation as conversation_lib
from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.engine.trainer import LLaVATrainer
from llava.models.llm import LlavaLlamaForCausalLM, LlavaQwen2ForCausalLM
from llava.utils import convert_expand_to_square
from llava.utils.checkpoint import safe_save_model_for_hf_trainer
from llava.utils.events import LOGGER
from llava.utils.ops import (
    find_all_linear_names, get_tokenize_len, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, process_anyres_image,
    process_highres_image, process_highres_image_crop_split, rank0_print, tokenizer_image_token,
)

torch.multiprocessing.set_sharing_strategy("file_system")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None


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
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target: torch.Tensor, tokenized_lens: List[int], speakers: List[str]) -> None:
    """Mask the targets based on tokenized lengths and speakers.

    Args:
        target (torch.Tensor): The target tensor to be masked.
        tokenized_lens (List[int]): A list of tokenized lengths for each round.
        speakers (List[str]): A list of speakers corresponding to each round.
    """
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2: cur_idx + tokenized_len] = IGNORE_INDEX
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
    begin_signal = "### "
    end_signal = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = begin_signal + from_str + ": " + sentence["value"] + end_signal
        if get_conversation:
            conversation += sentence["value"]
    conversation += begin_signal
    return conversation


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-13b-v1.5")

    version: Optional[str] = field(default="vicuna_v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_tunable_parts: Optional[str] = field(
        default=None,
        metadata={
            "help": "Could be 'mm_mlp_adapter', 'mm_vision_resampler', 'mm_vision_tower,mm_mlp_adapter,mm_language_model', 'mm_vision_tower,mm_mlp_adapter,mm_language_model', 'mm_mlp_adapter,mm_language_model'"}
    )
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)

    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)


@dataclass
class DataArguments:
    """Arguments pertaining to the data we are going to use for training and evaluation."""
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data."},
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)


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
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels,
                     attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            batch["image_sizes"] = [image[1] for image_list in images for image in image_list]
            batch["modalities"] = [image[2] for image_list in images for image in image_list]
            images = [image[0] for image_list in images for image in image_list]
            batch["images"] = images

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]

        return batch

    def pad_sequence(
            self,
            input_ids: Sequence[torch.Tensor],
            batch_first: bool,
            padding_value: Union[int, float],
    ) -> torch.Tensor:
        """Pad a list of variable‑length token‑ID tensors.

        This helper mirrors the behaviour of ``torch.nn.utils.rnn.pad_sequence`` but
        respects the ``tokenizer.padding_side`` attribute.  When the tokenizer
        pads on the *left*, each sequence is flipped before padding and the final
        batch tensor is flipped back so that the padding stays on the left side.

        Args:
            input_ids (Sequence): A sequence of variable‑length token‑ID tensors to be padded.
            batch_first (bool, optional): If ``True``, then the tensors must have the shape.
            padding_value (Union[int, float]): The value to use for padding.

        Returns:
            torch.Tensor: The padded tensor whose shape is determined by ``batch_first``.
        """
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Arguments pertaining to the training process."""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."},
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."},
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."},
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_variable_length: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "Use transformers attention implementation."},
    )


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
        rank0_print("Formatting inputs...Skip in lazy mode.")
        self.tokenizer: transformers.PreTrainedTokenizer = tokenizer
        self.list_data_dict: List[Dict[str, str]] = json.load(open(data_path, "r"))
        self.data_args: DataArguments = data_args

        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}.")
            self.data_args.dataset_paths = []
            for file_name in file_names:
                self.data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                self.data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy.")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy.
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            self.data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            with open(data_path, "r") as file:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}.")
        rank0_print("Formatting inputs...Skip in lazy mode.")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get an item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the input IDs, labels, and optionally images.
        """
        num_base_retries = 3

        for attempt_index in range(num_base_retries):
            try:
                sample = self._get_item(index)
                return sample
            except Exception as e:
                LOGGER.exception(f"Try [#{attempt_index}] failed to fetch sample {index}. {e}.")
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        next_index = 0
        for attempt_index in range(num_base_retries):
            try:
                next_index = min(index + 1, len(self.list_data_dict) - 1)
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                LOGGER.exception(f"Try other [#{attempt_index}] failed to fetch sample {next_index}. {e}.")
                pass

        try:
            sample = self._get_item(index)
            return sample
        except Exception as e:
            raise e

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.list_data_dict)

    def _get_item(self, index) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[index]
        if isinstance(index, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list."

        if "image" in sources[0]:
            image_file = self.list_data_dict[index]["image"]
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
                if len(image_file) > 1:
                    image = [self.process_image(f, "pad") for f in image_file]
                    image = [[img[0], img[1], "image"] for img in image]
            else:
                image = [self.process_image(image_file)]

            sources = preprocess_multimodal(deepcopy([e["conversations"] for e in sources]), self.data_args)
        # TODO: implements it.
        # elif "video" in sources[0]:
        #     video_file = self.list_data_dict[index]["video"]
        #     video_folder = self.data_args.video_folder
        #     video_file = os.path.join(video_folder, video_file)
        #     if not os.path.exists(video_file):
        #         LOGGER.error(f"File {video_file} not exist!")
        #
        #     try:
        #         if "shareVideoGPTV" in video_file:
        #             frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
        #             frame_files.sort()
        #
        #             if self.data_args.force_sample:
        #                 num_frames_to_sample = self.data_args.frames_upbound
        #             else:
        #                 num_frames_to_sample = 10
        #
        #             avg_fps = 2
        #             total_frames = len(frame_files)
        #             sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)
        #             frame_time = [i / 2 for i in sampled_indices]
        #             frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        #             video_time = total_frames / avg_fps
        #
        #             # Read and store the sampled frames.
        #             video = []
        #             for index in sampled_indices:
        #                 frame_path = frame_files[index]
        #                 try:
        #                     with Image.open(frame_path) as img:
        #                         frame = img.convert("RGB")
        #                         video.append(frame)
        #                 except IOError as e:
        #                     LOGGER.exception(f"Failed to read frame at path: {frame_path}. {e}.")
        #         else:
        #             video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file, self.data_args)
        #
        #         processor = self.data_args.image_processor
        #         image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
        #         if self.data_args.add_time_instruction:
        #             time_instruction = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        #             sources[0]["conversations"][0][
        #                 "value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruction}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
        #         image = [(image, video[0].size, "video")]
        #         sources = preprocess_multimodal(deepcopy([e["conversations"] for e in sources]), self.data_args)
        #     except Exception as e:
        #         LOGGER.exception(f"Failed to read video file: {video_file}. {e}.")
        #         return self._get_item(index + 1)
        else:
            sources = deepcopy([e["conversations"] for e in sources])

        # TODO: implements it.
        # has_image = ("image" in self.list_data_dict[index]) or ("video" in self.list_data_dict[index])
        has_image = "image" in self.list_data_dict[index]
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)

        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None

        if isinstance(index, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.list_data_dict[index]:
            data_dict["image"] = image
        # TODO: implements it.
        # elif "video" in self.list_data_dict[index]:
        #     data_dict["image"] = image
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = [
                (torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]), "text"),
            ]

        # prompt exist in the data.
        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = self.list_data_dict[index].get("id", index)
        return data_dict

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
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len)
            else:
                length_list.append(-cur_len)
        return length_list

    def process_image(
            self,
            image_file: str,
            overwrite_image_aspect_ratio: Optional[str] = None
    ) -> Tuple[Any, Any, str]:
        """Helper function to process an image.

        Args:
            image_file (str): The image file name.
            overwrite_image_aspect_ratio (Optional[str], optional): If provided, overwrite the default image aspect ratio. Defaults to ``None``.

        Returns:
            Tuple[Any, Any, str]: A tuple containing the processed image tensor, original image size, and modality type.
        """
        try:
            image = Image.open(str(Path(self.data_args.image_folder, image_file))).convert("RGB")
        except Exception as e:
            LOGGER.exception(f"Failed to open image {image_file}. {e}.")
            raise e

        if overwrite_image_aspect_ratio is not None:
            self.data_args.image_aspect_ratio = overwrite_image_aspect_ratio

        if self.data_args.image_aspect_ratio == "highres":
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif "anyres" in self.data_args.image_aspect_ratio:
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif self.data_args.image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif self.data_args.image_aspect_ratio == "pad":
            image = convert_expand_to_square(image, tuple(int(x * 255) for x in self.data_args.image_processor.image_mean))
            image = self.data_args.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = self.data_args.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image.size, "image"


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


def preprocess(
        sources: Sequence[List[str]],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
) -> Any:
    """Preprocess conversations for supervised.

    Args:
        sources (Sequence[List[str]]): A list of conversations, each conversation is a list of sentences.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.
        has_image (bool): Whether the conversations contain images.

    Returns:
        Any: A list of processed conversations with images replaced by image tokens.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.VICUNA_V1:
        return preprocess_vicuna_v1(sources, tokenizer, has_image)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA:
        return preprocess_llama(sources, tokenizer, has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image)

    # add end signal and concatenate together
    conversations = []
    header = None
    for source in sources:
        header = f"{conversation_lib.default_conversation.system_message}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    labels = deepcopy(input_ids)
    for label, source in zip(labels, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(label, tokenized_lens, speakers)

    return dict(
        input_ids=input_ids,
        labels=labels,
    )


def preprocess_multimodal(
        sources: Union[Sequence[List[str]], List[str]],
        data_args: DataArguments,
) -> Sequence[List[str]]:
    """Preprocess conversations for multimodal fine-tuning.

    Args:
        sources (Union[Sequence[List[str]], List[str]]): A list of conversations, each conversation is a list of sentences.
        data_args (DataArguments): The data arguments containing multimodal configurations.

    Returns:
        Any: A list of processed conversations with images replaced by image tokens.
    """
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            if num_image == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources


def preprocess_plain(
        sources: Dict[str, List[Dict[str, str]]],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict[str, List[Union[List[int], torch.Tensor]]]:
    """Preprocess conversations in plain style.

    Args:
        sources (Dict[str, List[Dict[str, str]]]): A list of conversations, each conversation is a list of sentences.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

    Returns:
        Dict[str, List[Union[List[int], torch.Tensor]]]: A dictionary containing the tokenized input IDs and labels.
    """
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)

    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = deepcopy(input_ids)
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

    # Apply prompt templates.
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human.
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations.
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

    # Mask targets: mask everything except assistant responses.
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


def preprocess_llama(
        sources: Dict[str, List[Dict[str, str]]],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
) -> Dict[str, List[Dict[str, str]]]:
    """Preprocess conversations in Llama2 style.

    Args:
        sources (Dict[str, List[Dict[str, str]]]): A list of conversations, each conversation is a list of sentences.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.
        has_image (bool): Whether the conversations contain images.

    Returns:
        Dict[str, List[Dict[str, str]]]: A dictionary containing the tokenized input IDs and labels.
    """
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates.
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human.
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations.
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

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA2

    # Mask targets: mask everything except assistant responses.
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


def preprocess_qwen(
        sources: Sequence[List[str]],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        system_message: str = "You are a helpful assistant."
) -> Any:
    """Preprocess conversations for Qwen1.5/2.

    Args:
        sources (Sequence[List[str]]): A list of conversations, each conversation is a list of sentences.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.
        has_image (bool): Whether the conversations contain images.
        system_message (str, optional): The system message to prepend to each conversation. Defaults to ``You are a helpful assistant.``.

    Returns:
        Any: A list of processed conversations with images replaced by image tokens.
    """
    # Unified name.
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer)

    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids
    unmask_tokens_index = [198, im_start, im_end]

    # Reset Qwen chat templates so that it won't include system message every time we apply.
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # input_ids: system+user+assistant.
    # labels: The corresponding label sequence (used in loss calculation).
    input_ids, labels = [], []
    # Iterate through each conversation.
    for i, source in enumerate(sources):
        # Filter the first system message.
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, label = [], []
        input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
        label += [IGNORE_INDEX] * len(input_id)  # Not involved in label learning.

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except Exception as e:  # noqa
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            conv = [{"role": role, "content": content}]  # Map "human"/"gpt" to "user"/"assistant".
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                label += [IGNORE_INDEX] * len(encode_id)
            else:  # role == "assistant".
                label += encode_id

        assert len(input_id) == len(label), f"{len(input_id)} != {len(label)}"
        for j, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_index:  # Structured control symbols. Example: <image>, <im_start>, <im_end>.
                label[j] = encode_id
            if encode_id == image_token_index:  # Vision token.
                input_id[j] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        labels.append(label)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=labels,
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
                quantization_config=transformers.BitsAndBytesConfig(
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
        if "qwen" in model_args.model_name_or_path.lower():
            model = LlavaQwen2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                local_files_only=True,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                local_files_only=True,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        if "qwen" in model_args.model_name_or_path.lower():
            model = transformers.Qwen2ForCausalLM.from_pretrained(
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
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # Load tokenizer.
    if "qwen" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
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
