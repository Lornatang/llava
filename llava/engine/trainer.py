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
from typing import Any, Iterator, List, Optional

import bitsandbytes
import torch
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from torch import nn
from torch.utils.data import Sampler
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from llava.utils.events import LOGGER

__all__ = [
    "LengthGroupedSampler", "LLaVATrainer", "get_mm_adapter_state_maybe_zero_3", "get_modality_length_grouped_indices", "maybe_zero_3",
    "split_to_even_chunks",
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

    def __iter__(self) -> Iterator[int]:
        """Yields indices for sampling.

        Returns:
            Iterator[int]: Iterator over sample indices.
        """
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    """LLaVATrainer class for LLaVA models, extending the base Trainer with custom functionality."""

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """Returns the training sampler, optionally grouping by modality length.

        Returns:
            Optional[torch.utils.data.Sampler]: The training sampler or None.
        """
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Creates and returns the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer instance.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in self.model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if
                            (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if
                            (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if
                            (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            # If using Adam8bit, register modules to optimize in fp32.
            if optimizer_cls.__name__ == "Adam8bit":
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in self.model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        LOGGER.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        LOGGER.debug(f"bitsandbytes: will optimize {module} in fp32")
                LOGGER.info(f"skipped: {skipped / 2 ** 20}M params")

        return self.optimizer

    def _save_checkpoint(self, model: nn.Module, trial: Any, metrics: Optional[dict] = None) -> None:
        """Saves a checkpoint during training.

        Args:
            model (nn.Module): The model being trained.
            trial (Any): Hyperparameter search trial object.
            metrics (Optional[dict], optional): Training metrics. Defaults to None.
        """
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = str(Path(run_dir, checkpoint_folder))

            # Only save Adapter.
            keys_to_match = ["mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, str(Path(output_dir, f"mm_projector.bin")))
        else:
            super(Trainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[dict] = None) -> None:
        """Saves the model and state dict.

        Args:
            output_dir (Optional[str], optional): Directory to save to. Defaults to None.
            state_dict (Optional[dict], optional): State dictionary to save. Defaults to None.
        """
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(Trainer, self)._save(output_dir, state_dict)


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


def get_modality_length_grouped_indices(
        lengths: List[int],
        batch_size: int,
        world_size: int,
        generator: Optional[torch.Generator] = None
) -> List[int]:
    """Groups indices by modality and length for batching.

    Args:
        lengths (List[int]): List of sample lengths (positive for multimodal, negative for language).
        batch_size (int): Batch size.
        world_size (int): Number of distributed workers.
        generator (Optional[torch.Generator], optional): Random generator. Defaults to None.

    Returns:
        List[int]: Grouped and shuffled indices.
    """
    assert all(l != 0 for l in lengths), "Should not have zero length."

    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)

    # multi-modal sample.
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    # language sample.
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
