# VGGT 缓存快速开始 🚀

## 一分钟上手

### 1️⃣ 预计算（一次性）

**方式 1：一键运行（推荐）**
```bash
# 编辑脚本修改路径
vim scripts/run_precompute.sh

# 运行
bash scripts/run_precompute.sh
```

**方式 2：环境变量**
```bash
VGGT_MODEL_PATH=/path/to/VGGT-1B \
DATA_ROOT=/path/to/train_data \
BATCH_SIZE=16 \
bash scripts/run_precompute.sh
```

**方式 3：直接调用 Python（高级）**
```bash
export PYTHONPATH=$PWD/src
python scripts/precompute_vggt_simple.py \
  --vggt_model_path /path/to/VGGT-1B \
  --data_root /path/to/train_data \
  --batch_size 16 \
  --skip_existing
```

MODEL_PATH="/public/home/vlabadmin/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"
VGGT_MODEL_PATH="/public/home/vlabadmin/.cache/modelscope/hub/models/facebook/VGGT-1B"

DATA_ROOT="/public/home/vlabadmin/dataset/VLN/JanusVLN_Trajectory_Data/trajectory_data/R2R-CE-640x480/train"
OUTPUT_DIR="/public/home/vlabadmin/dataset/NewJanusVLN/outputs/vln_h800_8gpu"
CACHE_DIR="./cache"

```bash
python scripts/precompute_vggt_features.py \
  --model_path /public/home/vlabadmin/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct \
  --vggt_model_path /public/home/vlabadmin/.cache/modelscope/hub/models/facebook/VGGT-1B \
  --data_root /public/home/vlabadmin/dataset/VLN/JanusVLN_Trajectory_Data/trajectory_data/R2R-CE-640x480/train \
  --batch_size 32 \
  --skip_existing
```



### 2️⃣ 训练（开启缓存）
```bash
export USE_VGGT_CACHE=true
bash scripts/train_h800.sh
```

## 核心变化

✅ **缓存文件位置**：`<图片路径>.vggt_cache.pt`（与图片放在一起）

✅ **无需单独缓存目录**：数据和缓存统一管理

✅ **预期加速**：77.86s/it → 15-20s/it（**3-5x 提速**）

## 示例

```
训练数据目录结构：

/data/train/
├── scene001/
│   ├── img_0001.jpg                    ← 原始图片
│   ├── img_0001.jpg.vggt_cache.pt     ← 缓存文件（新增）
│   ├── img_0002.jpg
│   ├── img_0002.jpg.vggt_cache.pt
│   └── ...
├── scene002/
│   ├── img_0001.jpg
│   ├── img_0001.jpg.vggt_cache.pt
│   └── ...
└── vggt_cache_manifest.json           ← 预计算统计信息
```

## 检查是否生效

训练开始时看到此消息即成功：
```
[INFO] VGGT feature cache enabled (loading from image directories)
```

详细文档：`docs/VGGT_CACHE_SIMPLIFIED.md`
