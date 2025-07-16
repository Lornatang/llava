MODEL_VERSION=vicuna-v1-3-7b
PROMPT_VERSION=plain

deepspeed llava/train/train_mem.py \
    --deepspeed ./tools/zero2.json \
    --model_name_or_path ./results/train/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path ./datasets/blip_laion_cc_sbu_558k/blip_laion_cc_sbu_558k.json \
    --image_folder ./datasets/blip_laion_cc_sbu_558k/images \
    --vision_tower ./results/pretrained_models/openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./results/train/llava-$MODEL_VERSION-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb