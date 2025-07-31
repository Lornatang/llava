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
import os
import bitsandbytes
import torch
from llava.utils.events import LOGGER
from llava.utils.ops import get_length_grouped_indices, get_mm_length_grouped_indices, get_mm_adapter_state_maybe_zero_3
from torch import nn
from torch.utils.data import Sampler
from transformers import Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

__all__ = [
    "LengthGroupedSampler", "LLaVATrainer",
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

    def __iter__(self) -> Iterator[List[Any]]:
        """Yields indices for sampling.

        Returns:
            Iterator[List[Any]]: Iterator over sample indices.
        """
        if self.group_by_modality:
            indices = get_mm_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    """LLaVATrainer class for LLaVA models, extending the base Trainer with custom functionality."""

    def _get_train_sampler(self, datasets: Any = None) -> Optional[torch.utils.data.Sampler]:
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
            else:  # Run this.
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
                        LOGGER.info(f"skipped {module}: {skipped / 2 ** 20}M params.")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        LOGGER.debug(f"bitsandbytes: will optimize {module} in fp32.")
                LOGGER.info(f"skipped: {skipped / 2 ** 20}M params.")

        return self.optimizer

    def _get_adapter_keys(self):
        keys = ["mm_projector", "vision_resampler"]
        if getattr(self.args, "use_im_start_end", False):
            keys.extend(["embed_tokens", "embed_in"])
        return keys

    def _save_checkpoint(self, model: nn.Module, trial: Any) -> None:
        """Saves a checkpoint during training.

        Args:
            model (nn.Module): The model being trained.
            trial (Any): Hyperparameter search trial object.
        """
        if getattr(self.args, "tune_mm_mlp_adapter", False):  # finetune.
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else: # pretrain.
            super(LLaVATrainer, self)._save_checkpoint(model, trial)

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[dict] = None) -> None:
        """Saves the model and state dict.

        Args:
            output_dir (Optional[str], optional): Directory to save to. Defaults to None.
            state_dict (Optional[dict], optional): State dictionary to save. Defaults to None.
        """
        if getattr(self.args, "tune_mm_mlp_adapter", False):  # finetune.
            return

        # pretrain.
        super(LLaVATrainer, self)._save(output_dir, state_dict)
