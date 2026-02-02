# VGGT 缓存快速开始 🚀

## 一分钟上手

### 1️⃣ 预计算（一次性）
```bash
python scripts/precompute_vggt_features.py \
  --model_path /path/to/Qwen2.5-VL-3B-Instruct \
  --vggt_model_path /path/to/VGGT-1B \
  --data_root /path/to/train_data \
  --batch_size 4 \
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
