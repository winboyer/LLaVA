#!/bin/bash

# export CUDA_VISIBLE_DEVICES=6
# export CUDA_VISIBLE_DEVICES=4
# export CUDA_VISIBLE_DEVICES="4,5,6,7" 

# --include=localhost:4,5,6,7
# --num_gpus=4 

deepspeed --include=localhost:0,1,2,3,4,5,6,7 \
    --master_port=25641 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3_v1.json \
    --model_name_or_path /root/jinyfeng/models/LLaVa/llava-v1.5-13b \
    --version plain \
    --data_path /root/jinyfeng/projects/papers_related/multimodel_gpt_v1/new_llava/LLaVA/playground/pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /root/jinyfeng/projects/papers_related/multimodel_gpt_v1/new_llava/LLaVA/playground/pretrain/images \
    --vision_tower /root/jinyfeng/projects/papers_related/multimodel_gpt_v1/tf_visiontower \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-13b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 800 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
