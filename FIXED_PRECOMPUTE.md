# ✅ 修复完成 - VGGT 缓存预计算脚本

## 问题原因

您遇到的导入错误是因为：
```python
ImportError: cannot import name 'Qwen2_5_VLForConditionalGeneration'
```

实际的类名是：`Qwen2_5_VLForConditionalGenerationForJanusVLN`

## 解决方案

我创建了一个**简化版预计算脚本**，直接使用 VGGT 模型，无需加载完整的 Qwen2.5-VL 模型。

---

## 🚀 立即使用（3 种方式）

### 方式 1：一键脚本（最简单）✨

```bash
# 1. 编辑脚本，修改这两个路径
vim scripts/run_precompute.sh

# 找到这两行，改成你的实际路径：
VGGT_MODEL_PATH="/public/home/vlabadmin/.cache/modelscope/hub/models/facebook/VGGT-1B"
DATA_ROOT="/public/home/vlabadmin/dataset/VLN/JanusVLN_Trajectory_Data/trajectory_data/R2R-CE-640x480/train"

# 2. 运行脚本
bash scripts/run_precompute.sh
```

### 方式 2：环境变量（灵活）

```bash
VGGT_MODEL_PATH=/public/home/vlabadmin/.cache/modelscope/hub/models/facebook/VGGT-1B \
DATA_ROOT=/public/home/vlabadmin/dataset/VLN/JanusVLN_Trajectory_Data/trajectory_data/R2R-CE-640x480/train \
BATCH_SIZE=32 \
bash scripts/run_precompute.sh
```

### 方式 3：直接调用 Python（您当前的方式）

```bash
export PYTHONPATH=$PWD/src

python scripts/precompute_vggt_simple.py \
  --vggt_model_path /public/home/vlabadmin/.cache/modelscope/hub/models/facebook/VGGT-1B \
  --data_root /public/home/vlabadmin/dataset/VLN/JanusVLN_Trajectory_Data/trajectory_data/R2R-CE-640x480/train \
  --batch_size 32 \
  --device cuda:0 \
  --skip_existing
```

---

## 📝 新增的文件

### 核心脚本

1. **`scripts/precompute_vggt_simple.py`** ⭐
   - 简化版预计算脚本
   - 无需 Qwen2.5-VL 模型
   - 直接使用 VGGT 模型处理图片
   - 已修复导入错误

2. **`scripts/run_precompute.sh`** ⭐
   - 一键运行脚本
   - 自动设置环境
   - 包含所有默认配置

3. **`scripts/precompute_vggt_cache.sh`**
   - 完整版 bash 脚本
   - 更多配置选项

### 文档

4. **`PRECOMPUTE_GUIDE.md`**
   - 详细使用指南
   - 故障排查
   - 性能优化建议

---

## 🎯 推荐使用方式

**对于您的环境，推荐使用方式 2（环境变量）：**

```bash
cd ~/dataset/NewJanusVLN

VGGT_MODEL_PATH=/public/home/vlabadmin/.cache/modelscope/hub/models/facebook/VGGT-1B \
DATA_ROOT=/public/home/vlabadmin/dataset/VLN/JanusVLN_Trajectory_Data/trajectory_data/R2R-CE-640x480/train \
BATCH_SIZE=32 \
DEVICE=cuda:0 \
bash scripts/run_precompute.sh
```

---

## ⚙️ 参数说明

| 参数 | 您的值 | 说明 |
|------|--------|------|
| `VGGT_MODEL_PATH` | `/public/home/vlabadmin/.cache/modelscope/hub/models/facebook/VGGT-1B` | VGGT 模型路径 |
| `DATA_ROOT` | `/public/home/vlabadmin/dataset/VLN/JanusVLN_Trajectory_Data/trajectory_data/R2R-CE-640x480/train` | 训练数据根目录 |
| `BATCH_SIZE` | `32` | 批量大小（您用的 32，可以保持）|
| `DEVICE` | `cuda:0` | GPU 设备 |
| `SKIP_EXISTING` | `true`（默认）| 跳过已存在的缓存 |

---

## ✅ 验证运行成功

### 预期输出

```
=========================================
  VGGT 缓存一键预计算
=========================================

配置信息：
  VGGT 模型: /public/home/vlabadmin/.cache/modelscope/hub/models/facebook/VGGT-1B
  数据目录:  /public/home/vlabadmin/dataset/VLN/.../train
  批量大小:  32
  GPU 设备:  cuda:0

[INFO] Loading VGGT model...
[INFO] VGGT model loaded successfully
[INFO] Processing 10000 images...
Processing: 100%|████████| 312/312 [01:30<00:00,  3.45it/s]

[SUCCESS] Processed 10000 images
[INFO] Cache files stored as: <image_path>.vggt_cache.pt
[INFO] Manifest saved to .../vggt_cache_manifest.json

[成功] 预计算完成！
```

### 检查缓存文件

```bash
# 查看生成的缓存文件数量
find /public/home/vlabadmin/dataset/VLN/JanusVLN_Trajectory_Data/trajectory_data/R2R-CE-640x480/train \
  -name "*.vggt_cache.pt" | wc -l

# 查看 manifest
cat /public/home/vlabadmin/dataset/VLN/JanusVLN_Trajectory_Data/trajectory_data/R2R-CE-640x480/train/vggt_cache_manifest.json
```

---

## 🎯 启用缓存训练

预计算完成后，启用缓存进行训练：

```bash
export USE_VGGT_CACHE=true
bash scripts/train_h800.sh
```

训练日志中应该看到：
```
[INFO] VGGT feature cache enabled (loading from image directories)
```

---

## 🔍 与原命令的对比

### 您原来的命令（有错误）：
```bash
python scripts/precompute_vggt_features.py \
  --model_path /public/home/vlabadmin/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct \  # ❌ 不再需要
  --vggt_model_path /public/home/vlabadmin/.cache/modelscope/hub/models/facebook/VGGT-1B \
  --data_root /public/home/vlabadmin/dataset/VLN/JanusVLN_Trajectory_Data/trajectory_data/R2R-CE-640x480/train \
  --batch_size 32 \
  --skip_existing
```

### 新命令（修复后）：
```bash
python scripts/precompute_vggt_simple.py \
  --vggt_model_path /public/home/vlabadmin/.cache/modelscope/hub/models/facebook/VGGT-1B \  # ✅ 只需 VGGT 模型
  --data_root /public/home/vlabadmin/dataset/VLN/JanusVLN_Trajectory_Data/trajectory_data/R2R-CE-640x480/train \
  --batch_size 32 \
  --skip_existing
```

**关键差异**：
- ✅ 无需 `--model_path`（不加载 Qwen2.5-VL）
- ✅ 使用 `precompute_vggt_simple.py`（新脚本）
- ✅ 修复了导入错误

---

## 📊 预期时间和空间

根据您的配置（batch_size=32，H800 GPU）：

- **R2R-CE 训练集**：约 10,000-20,000 张图片
- **预计时间**：30-60 分钟
- **磁盘空间**：10-100 GB（取决于图片数量）

---

## 🆘 如果还有问题

### 常见错误 1：找不到图片

```bash
# 检查数据目录结构
ls -R /public/home/vlabadmin/dataset/VLN/JanusVLN_Trajectory_Data/trajectory_data/R2R-CE-640x480/train | head -20
```

### 常见错误 2：VGGT 模型加载失败

```bash
# 检查 VGGT 模型文件
ls -lh /public/home/vlabadmin/.cache/modelscope/hub/models/facebook/VGGT-1B/
```

应该包含 `model.pth` 或类似的模型文件。

### 常见错误 3：GPU 显存不足

```bash
# 降低 batch size
BATCH_SIZE=8 bash scripts/run_precompute.sh
```

---

## 📚 更多文档

- **快速开始**：`VGGT_CACHE_QUICKSTART.md`
- **详细指南**：`PRECOMPUTE_GUIDE.md`
- **完整文档**：`docs/VGGT_CACHE_SIMPLIFIED.md`

---

**现在可以运行了！选择上面任意一种方式开始预计算。** 🚀
