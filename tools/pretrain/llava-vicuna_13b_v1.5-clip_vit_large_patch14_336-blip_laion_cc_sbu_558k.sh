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
    --model_name_or_path ./results/pretrained_models/lmsys/vicuna-13b-v1.5 \
    --version llava_plain \
    --data_path ./datasets/llava_pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./datasets/llava_pretrain/images \
    --vision_tower ./results/pretrained_models/openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --output_dir ./results/pretrain/llava-vicuna_13b_v1.5-clip_vit_large_patch14_336-blip_laion_cc_sbu_558k \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb