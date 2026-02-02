# VGGT 缓存预计算 - 快速使用指南

## 🚀 一键运行（推荐）

### 步骤 1：修改脚本中的路径

编辑 `scripts/run_precompute.sh`，修改以下两个路径：

```bash
# VGGT 模型路径
VGGT_MODEL_PATH="/your/path/to/VGGT-1B"

# 训练数据根目录
DATA_ROOT="/your/path/to/train_data"
```

### 步骤 2：运行脚本

```bash
bash scripts/run_precompute.sh
```

完成！缓存文件会自动生成在图片同目录。

---

## ⚙️ 自定义参数（可选）

如果不想修改脚本，可以通过环境变量覆盖：

```bash
# 自定义所有参数
VGGT_MODEL_PATH=/path/to/vggt \
DATA_ROOT=/path/to/data \
BATCH_SIZE=32 \
DEVICE=cuda:1 \
bash scripts/run_precompute.sh
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `VGGT_MODEL_PATH` | - | VGGT 模型目录 |
| `DATA_ROOT` | - | 训练数据根目录 |
| `BATCH_SIZE` | 16 | 批量大小（根据 GPU 显存调整）|
| `DEVICE` | cuda:0 | GPU 设备 |
| `SKIP_EXISTING` | true | 跳过已存在的缓存文件 |

---

## 📊 监控进度

预计算过程中会显示进度条：

```
[INFO] Processing 10000 images...
Processing: 100%|██████████| 625/625 [02:30<00:00,  4.17it/s]

[SUCCESS] Processed 10000 images
[INFO] Cache files stored as: <image_path>.vggt_cache.pt
```

---

## ✅ 验证缓存生成

### 检查缓存文件数量

```bash
find /path/to/train_data -name "*.vggt_cache.pt" | wc -l
```

### 查看 manifest 文件

```bash
cat /path/to/train_data/vggt_cache_manifest.json
```

输出示例：
```json
{
  "total_images": 10000,
  "processed_images": 10000,
  "vggt_model_path": "/path/to/VGGT-1B",
  "data_root": "/path/to/train_data",
  "cache_format": "<image_path>.vggt_cache.pt"
}
```

### 测试加载缓存

```bash
python -c "
import torch
cache_file = '/path/to/image.jpg.vggt_cache.pt'
data = torch.load(cache_file)
print('Keys:', list(data.keys()))
print('Features shape:', data['features'].shape)
"
```

预期输出：
```
Keys: ['features', 'path']
Features shape: torch.Size([1024, 256])  # 具体形状可能不同
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

## 🔧 故障排查

### 问题 1：找不到图片

**错误信息**：
```
[WARN] No images found!
```

**解决方法**：
1. 检查 `DATA_ROOT` 路径是否正确
2. 确保目录下有 `.jsonl` 文件或图片文件

### 问题 2：GPU 显存不足

**错误信息**：
```
RuntimeError: CUDA out of memory
```

**解决方法**：
减小批量大小：
```bash
BATCH_SIZE=4 bash scripts/run_precompute.sh
```

### 问题 3：VGGT 模型加载失败

**错误信息**：
```
[ERROR] VGGT_MODEL_PATH not found
```

**解决方法**：
1. 检查 VGGT 模型路径是否正确
2. 确保目录下有 `model.pth` 或其他模型文件

### 问题 4：导入错误

**错误信息**：
```
ImportError: cannot import name 'VGGT'
```

**解决方法**：
1. 确保在项目根目录运行脚本
2. 检查 `PYTHONPATH` 是否正确设置（脚本会自动设置）

---

## 📈 性能优化

### 根据 GPU 调整批量大小

| GPU 型号 | 显存 | 推荐 batch_size |
|---------|------|----------------|
| V100 | 32GB | 8-16 |
| A100 | 40GB | 16-32 |
| A100 | 80GB | 32-64 |
| H800 | 80GB | 32-64 |

### 使用多 GPU 预计算

在不同 GPU 上分别运行：

```bash
# GPU 0
DEVICE=cuda:0 DATA_ROOT=/data/split1 bash scripts/run_precompute.sh &

# GPU 1
DEVICE=cuda:1 DATA_ROOT=/data/split2 bash scripts/run_precompute.sh &

wait
```

### 断点续传

使用 `--skip_existing` 参数（默认开启），中断后重新运行会跳过已处理的图片：

```bash
# 第一次运行（中断）
bash scripts/run_precompute.sh

# 重新运行（自动跳过已完成）
bash scripts/run_precompute.sh
```

---

## 📝 完整流程示例

```bash
# 1. 进入项目目录
cd ~/dataset/NewJanusVLN

# 2. 激活环境
conda activate janusvln

# 3. 修改脚本中的路径（或使用环境变量）
vim scripts/run_precompute.sh

# 4. 运行预计算（预计 2-4 小时）
bash scripts/run_precompute.sh

# 5. 验证缓存生成
find /path/to/train_data -name "*.vggt_cache.pt" | wc -l

# 6. 启用缓存训练
export USE_VGGT_CACHE=true
bash scripts/train_h800.sh
```

---

## ⏱️ 预计时间

- **10,000 张图片**：约 30-60 分钟（取决于 GPU 和 batch size）
- **50,000 张图片**：约 2-4 小时
- **100,000 张图片**：约 4-8 小时

---

## 💾 磁盘空间

每张图片的缓存文件约 **1-5 MB**：

- 10,000 张图：约 10-50 GB
- 50,000 张图：约 50-250 GB
- 100,000 张图：约 100-500 GB

确保训练数据目录有足够空间！

---

## 📞 需要帮助？

如遇到问题：
1. 查看训练日志中的详细错误信息
2. 检查本文档的故障排查部分
3. 参考 `docs/VGGT_CACHE_SIMPLIFIED.md` 获取更多细节

---

**预计算完成后，训练速度将提升 3-5 倍！** 🚀
