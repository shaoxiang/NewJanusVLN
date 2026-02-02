# VGGT Feature Cache - 使用指南

## 快速开始

### 第一步：预计算 VGGT 特征（一次性，2-4小时）

```bash
# 在一个空闲的 GPU 上运行预计算脚本
python scripts/precompute_vggt_features.py \
  --model_path /path/to/Qwen2.5-VL-3B-Instruct \
  --vggt_model_path /path/to/VGGT-1B \
  --data_root /path/to/train_data \
  --cache_dir ./cache/vggt_features \
  --batch_size 1 \
  --num_workers 4 \
  --device cuda:0 \
  --verify

# 示例（使用你的实际路径）:
python scripts/precompute_vggt_features.py \
  --model_path /public/home/vlabadmin/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct \
  --vggt_model_path /public/home/vlabadmin/.cache/modelscope/hub/models/facebook/VGGT-1B \
  --data_root /public/home/vlabadmin/dataset/VLN/JanusVLN_Trajectory_Data/trajectory_data/R2R-CE-640x480/train \
  --cache_dir /public/home/vlabadmin/dataset/NewJanusVLN/cache/vggt_features \
  --batch_size 1 \
  --device cuda:0 \
  --verify
```

**预期输出**：
```
[INFO] Scanning data root: /path/to/train_data
[INFO] Found 10 JSONL files
[INFO] Total samples: 15000
[INFO] Unique images to process: 12000
[INFO] Model loaded successfully
Processing: 100%|██████████| 12000/12000 [2:30:00<00:00, 1.33it/s]
[SUCCESS] Processed 12000 images
[INFO] Cache directory: /path/to/cache/vggt_features
[INFO] Verification: 12000/12000 valid, 0 invalid
```

**磁盘占用估算**：
- 每张图像特征: ~2-5 MB
- 12,000 张图像: ~30-60 GB

---

### 第二步：启用缓存训练

#### 方式 A：环境变量（推荐）
```bash
export VGGT_CACHE_DIR=/path/to/cache/vggt_features
bash scripts/train_h800.sh
```

#### 方式 B：修改训练脚本
编辑 `scripts/train_h800.sh`，设置：
```bash
VGGT_CACHE_DIR="/path/to/cache/vggt_features"
```

---

## 验证加速效果

### 训练前监控
```bash
# 终端 1：启动训练
bash scripts/train_h800.sh

# 终端 2：实时监控
bash scripts/monitor_training.sh /path/to/outputs/vln_h800_8gpu/train_*.log
```

### 预期对比

| 配置 | 单步耗时 | Epoch耗时 | 加速比 |
|------|---------|----------|-------|
| **无缓存（原始）** | 77.86s/it | 620h (~26天) | 1.0x |
| **有缓存** | **15-20s/it** | **120-150h (~5-6天)** | **~4x** |
| **缓存 + ZeRO-2** | **12-15s/it** | **90-110h (~4-5天)** | **~5-6x** |

---

## 测试 ZeRO-2 + Offload（可选）

预计算完成后，显存压力大幅降低，可尝试：

```bash
# 修改 train_h800.sh
DS_CONFIG="scripts/zero2_offload.json"

# 可尝试增大 batch size
bash scripts/train_h800.sh
```

**预期效果**：
- 显存占用: 78GB → **45-55GB**
- 速度: 比 ZeRO-3 快 **1.5-2x**
- 可支持 `per_device_batch_size=2`

---

## 多节点训练（IB集群）

### 节点 0（主节点）
```bash
export MASTER_ADDR=192.168.1.100  # 主节点IP
export NODE_RANK=0
export NNODES=2
export VGGT_CACHE_DIR=/shared/cache/vggt_features  # 共享存储路径
bash scripts/train_2node_h800.sh
```

### 节点 1（工作节点）
```bash
export MASTER_ADDR=192.168.1.100
export NODE_RANK=1
export NNODES=2
export VGGT_CACHE_DIR=/shared/cache/vggt_features
bash scripts/train_2node_h800.sh
```

**关键配置**：
- `VGGT_CACHE_DIR` 必须在所有节点上可访问（NFS/共享存储）
- 调整 `NCCL_IB_HCA` 和 `NCCL_SOCKET_IFNAME` 匹配你的硬件

---

## 故障排查

### 1. 预计算脚本报错
**问题**: `ModuleNotFoundError: No module named 'qwen_vl'`
```bash
# 解决：确保在项目根目录运行
cd /path/to/NewJanusVLN
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python scripts/precompute_vggt_features.py ...
```

### 2. 缓存未生效
**检查**：训练日志中应有：
```
[INFO] VGGT feature cache enabled: /path/to/cache
```

**验证命中率**：
```python
# 在模型代码中临时加日志（modeling_qwen2_5_vl.py:2082 附近）
if use_cache and self.training:
    print(f"[DEBUG] Using cached features for sample {i}")
```

### 3. 预计算卡住
- 减小 `--batch_size` 到 1
- 检查图像文件是否损坏
- 查看 `nvidia-smi` 是否 OOM

### 4. 缓存路径权限问题
```bash
chmod -R 755 /path/to/cache/vggt_features
```

---

## 清理与重建

### 删除旧缓存
```bash
rm -rf /path/to/cache/vggt_features
```

### 增量更新（新增数据后）
```bash
python scripts/precompute_vggt_features.py \
  ... \
  --skip_existing  # 跳过已缓存的
```

---

## 高级优化

### 并行预计算（多 GPU）
```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_vggt_features.py \
  --data_root /path/to/train_data \
  --cache_dir /path/to/cache \
  --max_samples 6000 \
  ... &

# GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/precompute_vggt_features.py \
  --data_root /path/to/train_data \
  --cache_dir /path/to/cache \
  --max_samples -1 \
  --skip_existing \
  ... &

wait
```

### 验证缓存完整性
```bash
python scripts/precompute_vggt_features.py \
  --verify \
  --cache_dir /path/to/cache/vggt_features \
  --data_root /path/to/train_data
```

---

## 预期训练性能

基于你的硬件（8× H800）和当前日志：

| 阶段 | 配置 | 单步耗时 | Epoch时间 | 实际训练时间（3 epochs） |
|------|------|---------|----------|----------------------|
| 原始 | ZeRO-3, 无缓存 | 77.86s | 620h | **1860h (~78天)** |
| **优化后** | **ZeRO-3 + 缓存** | **18s** | **140h** | **420h (~17.5天)** |
| **最优** | **ZeRO-2 + 缓存 + 双节点** | **8s** | **60h** | **180h (~7.5天)** |

---

## 建议实施顺序

1. ✅ **今天**：运行预计算脚本（晚上挂着跑）
2. ✅ **明天**：启用缓存训练，验证加速效果
3. ✅ **后天**：测试 ZeRO-2，调优 batch size
4. ✅ **申请节点后**：多节点训练

**预计 3 天内在单节点完成首次训练，多节点后可缩短到 1 周内完成多次实验。**

---

需要帮助？
- 查看日志: `tail -f /path/to/outputs/train_*.log`
- 监控脚本: `bash scripts/monitor_training.sh <log_file>`
- GPU状态: `watch -n 1 nvidia-smi`
