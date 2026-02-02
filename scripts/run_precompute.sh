#!/bin/bash
#
# 一键预计算 VGGT 缓存
# 使用方法：直接运行此脚本，或设置环境变量覆盖默认路径
#
# 示例：
#   bash scripts/run_precompute.sh
#
# 或自定义路径：
#   VGGT_MODEL_PATH=/your/path bash scripts/run_precompute.sh

set -e

# ===== 配置区域：修改这里的路径 =====

# VGGT 模型路径
VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-/public/home/vlabadmin/.cache/modelscope/hub/models/facebook/VGGT-1B}"

# 训练数据根目录
DATA_ROOT="${DATA_ROOT:-/public/home/vlabadmin/dataset/VLN/JanusVLN_Trajectory_Data/trajectory_data/R2R-CE-640x480/train}"

# 处理设置
BATCH_SIZE="${BATCH_SIZE:-16}"          # 批量大小（根据显存调整）
DEVICE="${DEVICE:-cuda:0}"              # GPU 设备
SKIP_EXISTING="${SKIP_EXISTING:-true}"  # 跳过已存在的缓存

# ===== 以下无需修改 =====

echo "========================================="
echo "  VGGT 缓存一键预计算"
echo "========================================="
echo ""
echo "配置信息："
echo "  VGGT 模型: ${VGGT_MODEL_PATH}"
echo "  数据目录:  ${DATA_ROOT}"
echo "  批量大小:  ${BATCH_SIZE}"
echo "  GPU 设备:  ${DEVICE}"
echo "  跳过已存在: ${SKIP_EXISTING}"
echo ""
echo "========================================="

# 检查路径
if [[ ! -d "${VGGT_MODEL_PATH}" ]]; then
    echo "[错误] VGGT 模型路径不存在: ${VGGT_MODEL_PATH}"
    echo "请修改脚本中的 VGGT_MODEL_PATH 变量"
    exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[错误] 数据根目录不存在: ${DATA_ROOT}"
    echo "请修改脚本中的 DATA_ROOT 变量"
    exit 1
fi

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 设置 Python 路径
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# 进入项目根目录
cd "${PROJECT_ROOT}"

echo ""
echo "[开始] 预计算 VGGT 特征..."
echo ""

# 构建参数
ARGS=(
    "--vggt_model_path" "${VGGT_MODEL_PATH}"
    "--data_root" "${DATA_ROOT}"
    "--batch_size" "${BATCH_SIZE}"
    "--device" "${DEVICE}"
)

if [[ "${SKIP_EXISTING}" == "true" ]]; then
    ARGS+=("--skip_existing")
fi

# 运行预计算脚本
python scripts/precompute_vggt_simple.py "${ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "========================================="
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "[成功] 预计算完成！"
    echo ""
    echo "缓存文件已生成在图片同目录，格式："
    echo "  <图片路径>.vggt_cache.pt"
    echo ""
    echo "接下来启用缓存训练："
    echo "  export USE_VGGT_CACHE=true"
    echo "  bash scripts/train_h800.sh"
else
    echo "[失败] 预计算失败，退出码: ${EXIT_CODE}"
    echo ""
    echo "请检查错误信息并重试"
fi
echo "========================================="

exit ${EXIT_CODE}
