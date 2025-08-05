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
from datetime import timedelta
from functools import partial
from typing import Any, Optional

import bitsandbytes
import datasets
import torch
import torch.utils.data
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import GradientAccumulationPlugin
from torch import nn
from transformers import Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
)

from llava.data.sampler import LengthGroupedSampler
from llava.utils.events import LOGGER
from llava.utils.ops import rank0_print
from llava.utils.torch_utils import seed_worker

__all__ = [
    "LLaVATrainer",
]


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

    def create_accelerator_and_postprocess(self) -> None:
        grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
        grad_acc_kwargs["sync_with_dataloader"] = False
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        rank0_print("Setting NCCL timeout to INF to avoid running errors.")

        # create accelerator object.
        self.accelerator = Accelerator(
            deepspeed_plugin=self.args.deepspeed_plugin,
            gradient_accumulation_plugin=gradient_accumulation_plugin,
            kwargs_handlers=[accelerator_kwargs],
        )
        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag.
        self.gather_function = self.accelerator.gather_for_metrics

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher.
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        self.is_tp_enabled = getattr(self.accelerator.state, "tp_plugin", None) is not None

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get("limit_all_gathers", fsdp_plugin.limit_all_gathers)
            fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get("activation_checkpointing", fsdp_plugin.activation_checkpointing)
            if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                raise ValueError(
                    "The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg can't be set to True simultaneously. "
                    "Please use FSDP's activation_checkpointing logic when using FSDP.")

        if self.is_tp_enabled:
            LOGGER.warning(f"Not supporting TP training in LLaVA. ")
            pass

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
            lr_mapper = {}
            if self.args.mm_projector_lr is not None:
                lr_mapper["mm_projector"] = self.args.mm_projector_lr
            if self.args.mm_vision_tower_lr is not None:
                lr_mapper["vision_tower"] = self.args.mm_vision_tower_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [
                    name for name, _ in self.model.named_parameters()
                    if any(module_keyword in name for module_keyword in lr_mapper)
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in self.model.named_parameters()
                            if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in self.model.named_parameters()
                            if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [
                        name for name, _ in self.model.named_parameters()
                        if module_keyword in name
                    ]
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [
                                    p for n, p in self.model.named_parameters()
                                    if (n in decay_parameters and n in module_parameters and p.requires_grad)
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": lr,
                            },
                            {
                                "params": [
                                    p for n, p in self.model.named_parameters()
                                    if (n not in decay_parameters and n in module_parameters and p.requires_grad)
                                ],
                                "weight_decay": 0.0,
                                "lr": lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in self.model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in self.model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                skipped = 0
                for module in self.model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        LOGGER.info(f"Skipped {module}: {skipped / 2 ** 20}M params.")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        LOGGER.debug(f"Bitsandbytes: will optimize {module} in fp32.")
                LOGGER.info(f"Skipped: {skipped / 2 ** 20}M params.")

        return self.optimizer

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the training dataloader.

        Returns:
            torch.utils.data.DataLoader: The training dataloader.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if isinstance(self.train_dataset, datasets.Dataset):
            self.train_dataset = self._remove_unused_columns(self.train_dataset, description="training")
        else:
            self.data_collator = self._get_collator_with_removed_columns(self.data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(seed_worker, self.args.dataloader_num_workers, self.args.process_index)
            dataloader_params["prefetch_factor"] = self.args.dataloader_num_workers * 2 if self.args.dataloader_num_workers != 0 else None

        dataloader = self.accelerator.prepare(torch.utils.data.DataLoader(self.train_dataset, **dataloader_params))

        return dataloader
