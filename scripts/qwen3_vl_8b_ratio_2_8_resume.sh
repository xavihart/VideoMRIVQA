#!/bin/bash

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

# ======================
# Path Configuration
# ======================
MODEL_PATH="/storage/ice-shared/ae8803che/hxue/data/checkpoint/Qwen3-VL-8B-Instruct"
TRAIN_SCRIPT="/storage/ice-shared/ae8803che/hxue/data/eccv/VideoMRIVQA/qwen-vl-finetune/qwenvl/train/train_qwen.py"
DS_CONFIG="/storage/ice-shared/ae8803che/hxue/data/eccv/VideoMRIVQA/scripts/zero3.json"
OUTPUT_DIR="/storage/ice-shared/ae8803che/hxue/data/eccv/VideoMRIVQA/checkpoints/qwen3_vl_8b_ratio_2_8"
CACHE_DIR="/storage/ice-shared/ae8803che/hxue/data/eccv/VideoMRIVQA/cache"
RESUME_CHECKPOINT="/storage/ice-shared/ae8803che/hxue/data/eccv/VideoMRIVQA/checkpoints/qwen3_vl_8b_ratio_2_8/checkpoint-507"

# ======================
# Model Configuration
# ======================
DATASETS="brain_volume%80,knee_volume%80,brain_image%20,knee_image%20"

# ======================
# Training Hyperparameters
# ======================
# Target max_steps = 507 + 2000 = 2507
# Estimated epochs = 2507 / 169 ≈ 14.83
PYTHONPATH="/storage/ice-shared/ae8803che/hxue/data/eccv/VideoMRIVQA/qwen-vl-finetune:$PYTHONPATH" \
/storage/ice-shared/ae8803che/hxue/data/eccv/VideoMRIVQA/.venv/bin/torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         $TRAIN_SCRIPT \
         --model_name_or_path $MODEL_PATH \
         --tune_mm_llm True \
         --tune_mm_vision False \
         --tune_mm_mlp False \
         --dataset_use $DATASETS \
         --output_dir $OUTPUT_DIR \
         --cache_dir $CACHE_DIR \
         --bf16 \
         --per_device_train_batch_size 1 \
         --gradient_accumulation_steps 8 \
         --learning_rate 2e-5 \
         --mm_projector_lr 1e-5 \
         --vision_tower_lr 1e-6 \
         --optim adamw_torch \
         --model_max_length 4096 \
         --data_flatten True \
         --data_packing True \
         --max_pixels 451584 \
         --min_pixels 12544 \
         --video_fps 4 \
         --video_max_frames 16 \
         --video_min_frames 4 \
         --video_max_pixels 1304576 \
         --video_min_pixels 200704 \
         --max_steps 2507 \
         --num_train_epochs 15 \
         --warmup_ratio 0.03 \
         --lr_scheduler_type "cosine" \
         --weight_decay 0.01 \
         --logging_steps 1 \
         --report_to wandb \
         --run_name qwen3_vl_8b_ratio_2_8_resume_2000 \
         --save_steps 500 \
         --save_total_limit 3 \
         --lora_enable False \
         --resume_from_checkpoint $RESUME_CHECKPOINT \
         --deepspeed $DS_CONFIG
