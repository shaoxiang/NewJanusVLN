#!/usr/bin/env bash

# =========================================================
# JanusVLN training launcher (H800)
# - Adds strong self-checks and always prints useful logs
# - Avoids silent exit when paths/commands are missing
# =========================================================

set -euo pipefail

DEBUG=${DEBUG:-1}
if [[ "${DEBUG}" == "1" ]]; then
  set -x
fi

# Make python print immediately (helps when piping to tee)
export PYTHONUNBUFFERED=1

# Richer distributed/NCCL diagnostics (can be turned off by exporting these vars before running)
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}

# On any error, dump last lines of the log so it is never "silent"
trap 'rc=$?; echo "[FATAL] train_h800.sh failed with exit code ${rc}" >&2; if [[ -n "${LOG_FILE:-}" && -f "${LOG_FILE}" ]]; then echo "[FATAL] Last 200 lines of ${LOG_FILE}:" >&2; tail -n 200 "${LOG_FILE}" >&2 || true; fi; exit ${rc}' ERR

# =========================================================
# 1. Environment & Resources
# =========================================================
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export NCCL_NVLS_ENABLE=0

MASTER_ADDR="127.0.0.1"
# More portable than `shuf` (works on minimal envs)
MASTER_PORT=$((20000 + RANDOM % 10000))
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# Pick python
if command -v python >/dev/null 2>&1; then
  PY=python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "[FATAL] python/python3 not found in PATH" >&2
  exit 127
fi

echo "[INFO] PWD: ${PWD}"
echo "[INFO] PYTHON: $(${PY} -c 'import sys; print(sys.executable)' 2>/dev/null || echo ${PY})"
echo "[INFO] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NPROC_PER_NODE=${NPROC_PER_NODE}"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
# Preflight checks (fail fast with clear reason)
${PY} -c 'import torch; print("[INFO] torch", torch.__version__)' 2>/dev/null || { echo "[FATAL] PyTorch is not importable in this env (${PY})." >&2; exit 3; }
${PY} -c 'import deepspeed; print("[INFO] deepspeed", deepspeed.__version__)' 2>/dev/null || { echo "[FATAL] DeepSpeed is not importable in this env (${PY})." >&2; exit 3; }

# =========================================================
# 2. Paths (PLEASE customize for your cluster)
# =========================================================
MODEL_PATH="/public/home/vlabadmin/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"
VGGT_MODEL_PATH="/public/home/vlabadmin/.cache/modelscope/hub/models/facebook/VGGT-1B"

DATA_ROOT="/public/home/vlabadmin/dataset/VLN/JanusVLN_Trajectory_Data/trajectory_data/R2R-CE-640x480/train"
OUTPUT_DIR="/public/home/vlabadmin/dataset/NewJanusVLN/outputs/vln_h800_8gpu"
CACHE_DIR="./cache"

# DeepSpeed config (override via env, e.g. DS_CONFIG=scripts/zero2.json)
DS_CONFIG="${DS_CONFIG:-scripts/zero3.json}"

# Validate paths early (avoid silent exit)
if [[ ! -f "${DS_CONFIG}" ]]; then
  echo "[FATAL] DeepSpeed config not found: ${DS_CONFIG}" >&2
  exit 2
fi

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

# Make output dir; if not writable, fallback to workspace outputs
if ! mkdir -p "${OUTPUT_DIR}"; then
  echo "[WARN] Cannot create OUTPUT_DIR=${OUTPUT_DIR}. Falling back to ./outputs/vln_h800_8gpu" >&2
  OUTPUT_DIR="./outputs/vln_h800_8gpu"
  mkdir -p "${OUTPUT_DIR}"
fi

mkdir -p "${CACHE_DIR}"

LOG_FILE="${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] Logging to: ${LOG_FILE}"

# =========================================================
# 3. Training Hyperparameters
# =========================================================
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
NUM_EPOCHS=${NUM_EPOCHS:-3}
MAX_HISTORY_IMAGES=${MAX_HISTORY_IMAGES:-8}

echo "[INFO] Training config:"
echo "  Per-device batch: ${PER_DEVICE_BATCH}"
echo "  Gradient accumulation: ${GRAD_ACCUM_STEPS}"
echo "  Effective batch size: $((PER_DEVICE_BATCH * NPROC_PER_NODE * GRAD_ACCUM_STEPS))"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Max history images: ${MAX_HISTORY_IMAGES}"

# =========================================================
# 4. Launcher selection
# =========================================================
if command -v torchrun >/dev/null 2>&1; then
  LAUNCHER=(torchrun)
else
  echo "[WARN] torchrun not found, fallback to: ${PY} -m torch.distributed.run" >&2
  LAUNCHER=(${PY} -m torch.distributed.run)
fi

# =========================================================
# 4. Training command
# =========================================================

# NOTE: we tee logs; if anything fails, you will still see it in console.
"${LAUNCHER[@]}" \
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
  2>&1 | tee "${LOG_FILE}"
