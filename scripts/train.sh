#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$PWD/src

python -m qwen_vl.train.train_vln \
  --model_name_or_path /home/dell/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --vggt_model_path /home/dell/.cache/huggingface/hub/models--facebook--VGGT-1B/snapshots/860abec7937da0a4c03c41d3c269c366e82abdf9 \
  --train_data_root ./train_data \
  --output_dir ./outputs/vln \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --model_max_length 163840 \
  --num_train_epochs 1 \
  --subinstruction_weight 1.0 \
  --desc2d_weight 1.0 \
  --desc3d_weight 1.0 \
  --action_weight 1.0 \
  --bf16 True \
  --logging_steps 10 \
  --save_steps 1000
