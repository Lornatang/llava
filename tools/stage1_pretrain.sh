#!/bin/bash

# Environment Configuration.
export OMP_NUM_THREADS=8
export ACCELERATE_CPU_AFFINITY=1
export TOKENIZERS_PARALLELISM=false

# Distributed Configuration.
export NCCL_IB_DISABLE=1  # If has InfiniBand set to 0.
export NCCL_IB_GID_INDEX=3  # ibv_devinfo -v.
export NCCL_IB_HCA=mlx5_0  # Set to your InfiniBand HCA if available, otherwise can be omitted.
export NCCL_SOCKET_IFNAME=eno1  # Aligning RoCE over Ethernet with netdev
export NCCL_DEBUG=WARN  # Set to INFO for debugging, can be set to WARN or ERROR for production.
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 10000-19999 -n 1)

# Data Configuration.
DATA_PATH="./datasets/stage1_data.yaml"
IMAGE_FOLDER="./datasets/llava_pretrain/images"

# Train Configuration.
VERSION="llava_plain"
MODEL_PATH="./results/pretrained_models/lmsys/vicuna-13b-v1.5"
VISION_MODEL_PATH="./results/pretrained_models/openai/clip-vit-large-patch14-336"
RUN_NAME="llava-vicuna_13b_v1.5-clip_vit_large_patch14_336-stage1_data"
ATTN_IMPLEMENTATION="flash_attention_2"  # "flash_attention_2" or "flash_attention_3" or "sdpa"
TORCH_COMPILE_BACKEND="inductor"  # "inductor" or "eager"
DEEPSPEED_CONFIG="./tools/zero3.json"

# Training Hyperparameters.
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ./tools/train.py \
         --model_path ${MODEL_PATH} \
         --version ${VERSION} \
         --data_path ${DATA_PATH} \
         --image_folder ${IMAGE_FOLDER} \
         --vision_tower ${VISION_MODEL_PATH} \
         --mm_tunable_parts "mm_mlp_adapter" \
         --mm_vision_select_layer -2 \
         --mm_projector_type "mlp2x_gelu" \
         --mm_use_im_start_end False \
         --mm_use_im_patch_token False \
         --output_dir "./results/stage1_pretrain/${RUN_NAME}" \
         --num_train_epochs 1 \
         --per_device_train_batch_size 16 \
         --gradient_accumulation_steps 1 \
         --dataloader_drop_last True \
         --learning_rate 1e-3 \
         --lr_scheduler_type "cosine" \
         --weight_decay 0. \
         --warmup_ratio 0.03 \
         --save_strategy "steps" \
         --save_steps 1000 \
         --save_total_limit 1 \
         --logging_steps 1 \
         --bf16 True \
         --tf32 True \
         --model_max_length 8192 \
         --gradient_checkpointing True \
         --lazy_preprocess True \
         --report_to "wandb" \
         --run_name ${RUN_NAME} \
         --attn_implementation ${ATTN_IMPLEMENTATION} \
         --torch_compile True \
         --torch_compile_backend ${TORCH_COMPILE_BACKEND} \
         --deepspeed ${DEEPSPEED_CONFIG}
