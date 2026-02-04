#!/bin/bash

# =========================================================
# 1. 环境与资源配置 (Environment & Resources)
# =========================================================
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1
export NCCL_NVLS_ENABLE=0  # 如果遇到NCCL问题可尝试改为0，H800集群通常支持NVLink

# 自动获取端口，防止冲突
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
NPROC_PER_NODE=8  # H800 满配 8 卡

# =========================================================
# 2. 路径配置 (Paths)
# =========================================================
# 模型路径 (使用你提供的绝对路径)
MODEL_PATH="/public/home/vlabadmin/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"
VGGT_MODEL_PATH="/public/home/vlabadmin/.cache/modelscope/hub/models/facebook/VGGT-1B"

# 数据与输出路径
DATA_ROOT="/public/home/vlabadmin/dataset/VLN/JanusVLN_Trajectory_Data/trajectory_data/R2R-CE-640x480/train"
OUTPUT_DIR="/data/robot/humanoidvision/janus/NewJanusVLN/outputs/vln_h800_8gpu"
CACHE_DIR="./cache"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

# DeepSpeed 配置文件路径 (需确保该文件存在，见下文)
DS_CONFIG="scripts/zero3.json"

# =========================================================
# 3. 训练启动 (Training Command)
# =========================================================
# 备注：
# - 恢复了 max_pixels 等参数，这对 Qwen-VL 处理高分辨率图像非常重要
# - 恢复了 tune_mm_* 参数，明确微调 LLM 和 Projector，冻结 Vision Tower (根据原脚本逻辑)
# - 使用 --deepspeed 启用并行加速

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m qwen_vl.train.train_vln \
    --model_name_or_path "$MODEL_PATH" \
    --vggt_model_path "$VGGT_MODEL_PATH" \
    --train_data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --cache_dir "$CACHE_DIR" \
    --deepspeed "$DS_CONFIG" \
    --tune_mm_llm True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --bf16 True \
    --bf16_full_eval True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --mm_projector_lr 1e-5 \
    --vision_tower_lr 1e-6 \
    --model_max_length 163840 \
    --num_train_epochs 3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 2 \
    --evaluation_strategy "steps" \
    --eval_steps 30 \
    --eval_split_ratio 0.1 \
    --eval_num_samples 2 \
    --dataloader_num_workers 8 \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --group_by_modality_length True \
    --data_flatten False \
    --max_pixels $((576*28*28)) \
    --min_pixels $((16*28*28)) \
    --video_max_frames 8 \
    --video_min_frames 4 \
    --video_max_frame_pixels $((1664*28*28)) \
    --video_min_frame_pixels $((256*28*28)) \
    --log_level info \
    2>&1 | tee "${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"


export NNODES=2
export NODE_RANK=1
export MASTER_ADDR=173.0.87.2
export MASTER_PORT=29500

export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5

export MODEL_PATH="/public/home/vlabadmin/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"
export VGGT_MODEL_PATH="/public/home/vlabadmin/.cache/modelscope/hub/models/facebook/VGGT-1B"

export DATA_ROOT="/public/home/vlabadmin/dataset/VLN/JanusVLN_Trajectory_Data/trajectory_data/R2R-CE-640x480/train"
export OUTPUT_DIR="/public/home/vlabadmin/dataset/NewJanusVLN/outputs/vln_h800_8gpu"

bash scripts/train_2node_h800.sh