export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eno1
export NCCL_DEBUG=INFO
export ACCELERATE_CPU_AFFINITY=1
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=10000 \
    ./tools/train.py \
    --deepspeed ./tools/zero3.json \
    --model_name_or_path "./results/pretrained_models/lmsys/vicuna-13b-v1.5" \
    --version "vicuna_v1" \
    --data_path "./datasets/llava_finetune/llava_v1_5_mix665k.json" \
    --image_folder "./datasets/llava_finetune" \
    --video_folder "./datasets/llava_finetune" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower "./results/pretrained_models/openai/clip-vit-large-patch14-336" \
    --mm_projector_type "mlp2x_gelu" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio "anyres_max_9" \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type "spatial_unpad" \
    --output_dir "./results/finetune/llava-vicuna_13b_v1.5-clip_vit_large_patch14_336-llava_v1_5_mix665k" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "wandb" \
    --run_name "llava-vicuna_13b_v1.5-clip_vit_large_patch14_336-llava_v1_5_mix665k" \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32