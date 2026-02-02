# VGGT 缓存优化指南（简化版）

## 核心改动

VGGT 缓存文件现在**直接存储在图片所在目录**，与图片放在一起，无需单独的缓存目录。

### 缓存文件位置

对于每张图片，缓存文件的命名规则：
```
<原始图片路径>.vggt_cache.pt
```

**示例**：
```
/data/train/scene001/img_0001.jpg          # 原始图片
/data/train/scene001/img_0001.jpg.vggt_cache.pt  # 缓存文件
```

---

## 使用步骤

### 步骤 1：预计算 VGGT 特征

运行预计算脚本（一次性任务，约 2-4 小时）：

```bash
python scripts/precompute_vggt_features.py \
  --model_path /path/to/Qwen2.5-VL-3B-Instruct \
  --vggt_model_path /path/to/VGGT-1B \
  --data_root /path/to/train_data \
  --batch_size 4 \
  --device cuda:0 \
  --skip_existing \
  --verify
```

**注意**：
- 不再需要 `--cache_dir` 参数
- 缓存文件会自动生成在每张图片所在目录
- 可以多次运行，使用 `--skip_existing` 跳过已处理的图片

### 步骤 2：启用缓存训练

**单节点训练**：
```bash
export USE_VGGT_CACHE=true
bash scripts/train_h800.sh
```

**双节点训练**：
```bash
export USE_VGGT_CACHE=true
bash scripts/train_2node_h800.sh
```

---

## 优势

### 1. 数据组织更清晰
- 缓存文件与图片放在一起，便于管理
- 无需维护单独的缓存目录
- 便于备份和迁移（图片+缓存一起复制）

### 2. 避免哈希计算
- 不再需要计算图片的 SHA256 hash
- 加载缓存时直接拼接路径即可
- 减少 IO 开销

### 3. 更易理解和调试
- 缓存文件位置直观
- 可以直接查看每张图片是否有对应缓存
- 便于手动清理或重新生成特定图片的缓存

---

## 文件变更说明

### 1. 预计算脚本（`scripts/precompute_vggt_features.py`）

**主要改动**：
- 移除 `--cache_dir` 参数
- 缓存文件生成在图片路径 + `.vggt_cache.pt`
- manifest 文件生成在 `data_root` 下

### 2. 数据加载器（`src/qwen_vl/data/vln_data.py`）

**主要改动**：
```python
def _load_cached_vggt_features(self, image_path: str):
    """从图片同目录加载缓存"""
    cache_path = f"{image_path}.vggt_cache.pt"
    if os.path.exists(cache_path):
        data = torch.load(cache_path, map_location="cpu")
        return data["features"]
    return None
```

### 3. 训练参数（`src/qwen_vl/train/argument.py`）

**主要改动**：
- 移除 `vggt_cache_dir` 参数
- 新增 `use_vggt_cache` 布尔参数

### 4. 训练脚本（`scripts/train_h800.sh`, `scripts/train_2node_h800.sh`）

**主要改动**：
- 使用 `USE_VGGT_CACHE=true` 环境变量
- 移除 `VGGT_CACHE_DIR` 环境变量

---

## 预期加速效果

- **单步时间**：77.86s → 15-20s（**3-5x 加速**）
- **单 epoch 时间**：620h → 144h（使用缓存）
- **双节点 epoch 时间**：144h → 72h

---

## 常见问题

### Q: 缓存文件会占用多少磁盘空间？

A: 每张图片的缓存文件约 1-5MB（取决于 VGGT 输出维度）。如果有 10,000 张训练图片，总共约 10-50GB。

### Q: 如何重新生成某张图片的缓存？

A: 直接删除对应的 `.vggt_cache.pt` 文件，然后重新运行预计算脚本（使用 `--skip_existing`）。

### Q: 缓存文件如何与图片一起备份？

A: 直接复制整个数据目录即可，缓存文件会随图片一起复制。

### Q: 如果图片更新了怎么办？

A: 删除对应的缓存文件，重新运行预计算脚本。

### Q: 多节点训练如何共享缓存？

A: 只要所有节点都能访问到相同的数据目录（包括图片和缓存文件），就可以自动共享缓存。使用 NFS 或共享存储即可。

---

## 验证缓存是否生效

训练开始时查看日志：

```
[INFO] VGGT feature cache enabled (loading from image directories)
```

如果看到大量警告：
```
[WARN] Failed to load cache /path/to/image.jpg.vggt_cache.pt
```

说明缓存文件不存在或损坏，需要运行预计算脚本。
