#!/bin/bash
#
# 8 GPU 并行预计算 VGGT 缓存
# 自动分配图片到 8 个 GPU 并行处理，大幅加速预计算过程
#
# 使用方法：
#   bash scripts/precompute_8gpu.sh

set -e

# ===== 配置区域 =====

# VGGT 模型路径
VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-/public/home/vlabadmin/.cache/modelscope/hub/models/facebook/VGGT-1B}"

# 训练数据根目录
DATA_ROOT="${DATA_ROOT:-/public/home/vlabadmin/dataset/VLN/JanusVLN_Trajectory_Data/trajectory_data/R2R-CE-640x480/train}"

# GPU 数量
NUM_GPUS="${NUM_GPUS:-8}"

# 每个 GPU 的批量大小
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-2048}"

# ===== 以下无需修改 =====

echo "========================================="
echo "  8 GPU 并行 VGGT 缓存预计算"
echo "========================================="
echo ""
echo "配置信息："
echo "  VGGT 模型:  ${VGGT_MODEL_PATH}"
echo "  数据目录:   ${DATA_ROOT}"
echo "  GPU 数量:   ${NUM_GPUS}"
echo "  每GPU批量:  ${BATCH_SIZE_PER_GPU}"
echo ""
echo "========================================="

# 验证路径
if [[ ! -d "${VGGT_MODEL_PATH}" ]]; then
    echo "[错误] VGGT 模型路径不存在: ${VGGT_MODEL_PATH}"
    exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[错误] 数据根目录不存在: ${DATA_ROOT}"
    exit 1
fi

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 设置 Python 路径
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
cd "${PROJECT_ROOT}"

# ===== 步骤 1: 收集所有图片路径 =====

echo ""
echo "[步骤 1/3] 收集图片路径..."

TMP_DIR="${PROJECT_ROOT}/.tmp_precompute"
mkdir -p "${TMP_DIR}"

IMAGE_LIST_FILE="${TMP_DIR}/all_images.txt"

# 运行 Python 脚本收集图片（不处理）
python3 -c "
import json
import os
import sys
from pathlib import Path

data_root = '${DATA_ROOT}'
output_file = '${IMAGE_LIST_FILE}'

print(f'[INFO] 扫描数据目录: {data_root}')

# 查找所有 JSONL 文件
jsonl_files = list(Path(data_root).rglob('*.jsonl'))

if not jsonl_files:
    print('[WARN] 未找到 JSONL 文件，直接扫描图片...')
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(Path(data_root).rglob(ext))
    all_images = sorted(set(str(p) for p in image_paths))
else:
    print(f'[INFO] 找到 {len(jsonl_files)} 个 JSONL 文件')
    
    all_images = set()
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    images = []
                    if 'images' in sample:
                        images = sample['images'] if isinstance(sample['images'], list) else [sample['images']]
                    elif 'image' in sample:
                        images = [sample['image']] if isinstance(sample['image'], str) else sample['image']
                    elif 'history_images' in sample and 'current_image' in sample:
                        images = list(sample['history_images']) + [sample['current_image']]
                    
                    data_path = sample.get('data_path', '')
                    for img_rel in images:
                        if os.path.isabs(img_rel):
                            img_full = img_rel
                        else:
                            img_full = os.path.join(data_root, data_path, img_rel)
                            if not os.path.exists(img_full):
                                img_full = os.path.join(os.path.dirname(jsonl_file), img_rel)
                        
                        if os.path.exists(img_full):
                            all_images.add(img_full)
                except:
                    pass
    
    all_images = sorted(all_images)

print(f'[INFO] 找到 {len(all_images)} 张图片')

# 写入文件
with open(output_file, 'w') as f:
    for img in all_images:
        f.write(img + '\n')

print(f'[INFO] 图片列表已保存到: {output_file}')
"

if [[ ! -f "${IMAGE_LIST_FILE}" ]]; then
    echo "[错误] 无法生成图片列表"
    exit 1
fi

TOTAL_IMAGES=$(wc -l < "${IMAGE_LIST_FILE}")
echo "[INFO] 找到 ${TOTAL_IMAGES} 张图片"

# ===== 步骤 2: 分割图片列表 =====

echo ""
echo "[步骤 2/3] 分割图片到 ${NUM_GPUS} 个 GPU..."

IMAGES_PER_GPU=$((TOTAL_IMAGES / NUM_GPUS + 1))

for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    START_LINE=$((gpu_id * IMAGES_PER_GPU + 1))
    END_LINE=$(((gpu_id + 1) * IMAGES_PER_GPU))
    
    GPU_IMAGE_LIST="${TMP_DIR}/images_gpu${gpu_id}.txt"
    sed -n "${START_LINE},${END_LINE}p" "${IMAGE_LIST_FILE}" > "${GPU_IMAGE_LIST}"
    
    GPU_IMAGES=$(wc -l < "${GPU_IMAGE_LIST}")
    echo "[INFO] GPU ${gpu_id}: ${GPU_IMAGES} 张图片"
done

# ===== 步骤 3: 并行处理 =====

echo ""
echo "[步骤 3/3] 启动 ${NUM_GPUS} GPU 并行处理..."
echo ""

# 创建日志目录
LOG_DIR="${PROJECT_ROOT}/outputs/precompute_logs"
mkdir -p "${LOG_DIR}"

# 启动所有 GPU 进程
PIDS=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_IMAGE_LIST="${TMP_DIR}/images_gpu${gpu_id}.txt"
    LOG_FILE="${LOG_DIR}/gpu${gpu_id}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "[启动] GPU ${gpu_id} -> ${LOG_FILE}"
    
    # 在后台运行
    CUDA_VISIBLE_DEVICES=${gpu_id} python scripts/precompute_vggt_simple.py \
        --vggt_model_path "${VGGT_MODEL_PATH}" \
        --data_root "${DATA_ROOT}" \
        --batch_size "${BATCH_SIZE_PER_GPU}" \
        --device "cuda:0" \
        --image_list_file "${GPU_IMAGE_LIST}" \
        > "${LOG_FILE}" 2>&1 &
    
    PIDS+=($!)
    
    # 短暂延迟避免同时启动导致竞争
    sleep 2
done

echo ""
echo "[INFO] 所有 GPU 已启动，等待完成..."
echo "[INFO] 实时监控命令: tail -f ${LOG_DIR}/gpu*.log"
echo ""

# 等待所有进程完成
SUCCESS_COUNT=0
FAILED_COUNT=0

for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    gpu_id=$i
    
    echo "[等待] GPU ${gpu_id} (PID: ${pid})..."
    
    if wait ${pid}; then
        echo "[完成] GPU ${gpu_id} 成功完成"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "[失败] GPU ${gpu_id} 处理失败"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

# ===== 步骤 4: 生成总结 =====

echo ""
echo "========================================="
echo "  预计算完成"
echo "========================================="
echo ""
echo "结果统计："
echo "  总图片数:   ${TOTAL_IMAGES}"
echo "  成功 GPU:   ${SUCCESS_COUNT} / ${NUM_GPUS}"
echo "  失败 GPU:   ${FAILED_COUNT} / ${NUM_GPUS}"
echo ""

# 统计缓存文件
CACHE_FILES=$(find "${DATA_ROOT}" -name "*.vggt_cache.pt" 2>/dev/null | wc -l)
echo "  缓存文件数: ${CACHE_FILES}"
echo ""

if [[ ${FAILED_COUNT} -eq 0 ]]; then
    echo "[成功] 所有 GPU 处理完成！"
    echo ""
    echo "提示：当前训练代码已移除 VGGT 缓存读取，这些 .vggt_cache.pt 不会被训练使用。"
else
    echo "[警告] 部分 GPU 处理失败，请检查日志："
    echo "  ${LOG_DIR}/"
fi

echo "========================================="

# 清理临时文件（可选）
# rm -rf "${TMP_DIR}"

exit 0
