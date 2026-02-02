#!/bin/bash
#
# VGGT Feature Precomputation Script
# Generates .vggt_cache.pt files next to each training image
#
# Usage:
#   bash scripts/precompute_vggt_cache.sh

set -e  # Exit on error

# =========================================================
# Configuration
# =========================================================

# VGGT model path (modify to your actual path)
VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-/public/home/vlabadmin/.cache/modelscope/hub/models/facebook/VGGT-1B}"

# Data root (modify to your training data directory)
DATA_ROOT="${DATA_ROOT:-/public/home/vlabadmin/dataset/VLN/JanusVLN_Trajectory_Data/trajectory_data/R2R-CE-640x480/train}"

# Processing settings
BATCH_SIZE="${BATCH_SIZE:-8}"           # Adjust based on GPU memory
DEVICE="${DEVICE:-cuda:0}"              # GPU device
NUM_WORKERS="${NUM_WORKERS:-4}"         # Data loading workers
MAX_SAMPLES="${MAX_SAMPLES:--1}"        # -1 for all samples

# Script settings
SKIP_EXISTING="${SKIP_EXISTING:-true}"  # Skip already cached images
VERIFY="${VERIFY:-true}"                # Verify cache after generation

# =========================================================
# Validation
# =========================================================

echo "========================================"
echo "VGGT Feature Precomputation"
echo "========================================"
echo "VGGT model path:  ${VGGT_MODEL_PATH}"
echo "Data root:        ${DATA_ROOT}"
echo "Batch size:       ${BATCH_SIZE}"
echo "Device:           ${DEVICE}"
echo "Skip existing:    ${SKIP_EXISTING}"
echo "========================================"

# Validate paths
if [[ ! -d "${VGGT_MODEL_PATH}" ]]; then
  echo "[ERROR] VGGT_MODEL_PATH not found: ${VGGT_MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] DATA_ROOT not found: ${DATA_ROOT}" >&2
  exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# =========================================================
# Run Precomputation
# =========================================================

echo ""
echo "[INFO] Starting precomputation..."
echo "[INFO] Working directory: ${PROJECT_ROOT}"
echo ""

cd "${PROJECT_ROOT}"

# Build arguments
ARGS=(
  "--vggt_model_path" "${VGGT_MODEL_PATH}"
  "--data_root" "${DATA_ROOT}"
  "--batch_size" "${BATCH_SIZE}"
  "--device" "${DEVICE}"
  "--max_samples" "${MAX_SAMPLES}"
)

if [[ "${SKIP_EXISTING}" == "true" ]]; then
  ARGS+=("--skip_existing")
fi

# Run Python script (using simplified version)
python scripts/precompute_vggt_simple.py "${ARGS[@]}"

EXIT_CODE=$?

# =========================================================
# Summary
# =========================================================

echo ""
echo "========================================"
if [[ ${EXIT_CODE} -eq 0 ]]; then
  echo "[SUCCESS] Precomputation completed!"
  echo ""
  echo "Cache files are stored as: <image_path>.vggt_cache.pt"
  echo "Manifest file: ${DATA_ROOT}/vggt_cache_manifest.json"
  echo ""
  echo "To use cached features during training:"
  echo "  export USE_VGGT_CACHE=true"
  echo "  bash scripts/train_h800.sh"
else
  echo "[FAILED] Precomputation failed with exit code ${EXIT_CODE}"
fi
echo "========================================"

exit ${EXIT_CODE}
