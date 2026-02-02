#!/bin/bash
#================================================================
# Multi-Node Training Script for JanusVLN on H800 Cluster
#================================================================
# This script supports both single-node and multi-node training
# with InfiniBand (IB) or Ethernet networking.
#
# Usage:
#   Single node (8 GPUs):
#     bash scripts/train_2node_h800.sh
#
#   Multi-node (Node 0 - Master):
#     export MASTER_ADDR=192.168.1.100  # Master node IP
#     export NODE_RANK=0
#     export NNODES=2
#     bash scripts/train_2node_h800.sh
#
#   Multi-node (Node 1 - Worker):
#     export MASTER_ADDR=192.168.1.100  # Master node IP
#     export NODE_RANK=1
#     export NNODES=2
#     bash scripts/train_2node_h800.sh
#================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail

# =========================================================
# 1. Environment Configuration
# =========================================================

# Multi-node setup (override with environment variables)
export NNODES=${NNODES:-1}                      # Total number of nodes
export NODE_RANK=${NODE_RANK:-0}                # Current node rank (0 for master)
export MASTER_ADDR=${MASTER_ADDR:-localhost}    # Master node address
export MASTER_PORT=${MASTER_PORT:-29500}        # Master port

# Per-node GPUs
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# NCCL configuration for multi-node training
if [[ $NNODES -gt 1 ]]; then
  echo "[INFO] Multi-node training enabled: $NNODES nodes"
  echo "[INFO] Current node rank: $NODE_RANK"
  echo "[INFO] Master address: $MASTER_ADDR:$MASTER_PORT"
  
  # NCCL optimization for InfiniBand
  export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}        # Enable IB (set to 1 for Ethernet-only)
  export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5}             # IB device (adjust per hardware)
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0} # Network interface
  export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-5}  # GPU Direct RDMA
  export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}    # RoCE mode
  export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}       # IB timeout (default: 18)
  export NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-7}    # IB retry count
  
  # Advanced NCCL tuning for H800
  export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-0}          # Disable cross-NIC for better stability
  export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}        # P2P level
  export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}      # Enable shared memory
  export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-8388608}      # Buffer size (8MB)
  export NCCL_NTHREADS=${NCCL_NTHREADS:-640}          # NCCL threads for H800
  
  # NCCL debugging (comment out for production)
  export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
  export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-ALL}
else
  echo "[INFO] Single-node training: $NPROC_PER_NODE GPUs"
fi

# Distributed training environment
export PYTHONUNBUFFERED=1
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0

# Python and paths
PY=${PY:-python}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(dirname "$SCRIPT_DIR")"
cd "$WORK_DIR"

# =========================================================
# 2. Path Configuration
# =========================================================

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-VL-3B-Instruct}"
VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-/path/to/VGGT-1B}"
DATA_ROOT="${DATA_ROOT:-/path/to/train_data}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/vln_2node_h800}"
CACHE_DIR="${CACHE_DIR:-./cache}"
DS_CONFIG="${DS_CONFIG:-./scripts/zero3.json}"

# Validate paths
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[FATAL] MODEL_PATH not found: ${MODEL_PATH}" >&2
  exit 2
fi

if [[ ! -d "${VGGT_MODEL_PATH}" ]]; then
  echo "[FATAL] VGGT_MODEL_PATH not found: ${VGGT_MODEL_PATH}" >&2
  exit 2
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[FATAL] DATA_ROOT not found: ${DATA_ROOT}" >&2
  exit 2
fi

# Create output dir
if ! mkdir -p "${OUTPUT_DIR}"; then
  echo "[WARN] Cannot create OUTPUT_DIR=${OUTPUT_DIR}. Falling back to ./outputs/vln_2node_h800" >&2
  OUTPUT_DIR="./outputs/vln_2node_h800"
  mkdir -p "${OUTPUT_DIR}"
fi

mkdir -p "${CACHE_DIR}"

LOG_FILE="${OUTPUT_DIR}/train_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] Logging to: ${LOG_FILE}"

# =========================================================
# 3. Training Hyperparameters
# =========================================================

# Effective batch size = per_device_batch * num_gpus * num_nodes * grad_accum
# Example: 1 * 8 * 2 * 8 = 128
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
NUM_EPOCHS=${NUM_EPOCHS:-3}
MAX_HISTORY_IMAGES=${MAX_HISTORY_IMAGES:-8}

echo "[INFO] Training config:"
echo "  Nodes: $NNODES x $NPROC_PER_NODE GPUs = $((NNODES * NPROC_PER_NODE)) total GPUs"
echo "  Per-device batch: $PER_DEVICE_BATCH"
echo "  Gradient accumulation: $GRAD_ACCUM_STEPS"
echo "  Effective batch size: $((PER_DEVICE_BATCH * NPROC_PER_NODE * NNODES * GRAD_ACCUM_STEPS))"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Max history images: $MAX_HISTORY_IMAGES"

# =========================================================
# 4. Launcher Selection
# =========================================================
if command -v torchrun >/dev/null 2>&1; then
  LAUNCHER=(torchrun)
else
  echo "[WARN] torchrun not found, fallback to: ${PY} -m torch.distributed.run" >&2
  LAUNCHER=(${PY} -m torch.distributed.run)
fi

# =========================================================
# 5. Training Command
# =========================================================

echo "=" | head -c 80 && echo
echo "[START] Training begins at $(date)"
echo "=" | head -c 80 && echo

"${LAUNCHER[@]}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m qwen_vl.train.train_vln \
  --model_name_or_path "${MODEL_PATH}" \
  --vggt_model_path "${VGGT_MODEL_PATH}" \
  --train_data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --deepspeed "${DS_CONFIG}" \
  --tune_mm_llm True \
  --tune_mm_vision False \
  --tune_mm_mlp True \
  --bf16 True \
  --bf16_full_eval True \
  --per_device_train_batch_size ${PER_DEVICE_BATCH} \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
  --learning_rate ${LEARNING_RATE} \
  --mm_projector_lr 1e-5 \
  --vision_tower_lr 1e-6 \
  --model_max_length 163840 \
  --num_train_epochs ${NUM_EPOCHS} \
  --max_history_images ${MAX_HISTORY_IMAGES} \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --save_steps 500 \
  --save_total_limit 2 \
  --evaluation_strategy no \
  --eval_steps 300000 \
  --eval_split_ratio 0.0 \
  --eval_num_samples 2 \
  --dataloader_num_workers 16 \
  --dataloader_pin_memory True \
  --dataloader_prefetch_factor 4 \
  --gradient_checkpointing True \
  --ddp_timeout 7200 \
  --ddp_find_unused_parameters False \
  --report_to "tensorboard" \
  --group_by_modality_length True \
  --data_flatten False \
  --max_pixels $((576*28*28)) \
  --min_pixels $((16*28*28)) \
  --video_max_frames 8 \
  --subinstruction_weight 1.0 \
  --desc2d_weight 1.0 \
  --desc3d_weight 1.0 \
  --action_weight 2.0 \
  --distill_loss_weight 1.0 \
  2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo "=" | head -c 80 && echo
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "[SUCCESS] Training completed at $(date)"
else
  echo "[FAILED] Training exited with code $EXIT_CODE at $(date)"
  echo "[DEBUG] Last 50 lines of log:"
  tail -n 50 "${LOG_FILE}"
fi
echo "=" | head -c 80 && echo

exit $EXIT_CODE
