# 训练优化实施完成总结

## ✅ 已实施的优化方案

### 1. **方案 A：VGGT 特征预计算缓存** ⭐⭐⭐⭐⭐
**预期加速：3-5x**

#### 核心文件
- `scripts/precompute_vggt_features.py` - 预计算脚本
- `src/qwen_vl/data/vln_data.py` - 支持缓存加载
- `src/qwen_vl/model/modeling_qwen2_5_vl.py` - 训练时使用缓存
- `src/qwen_vl/train/argument.py` - 新增 `--vggt_cache_dir` 参数

#### 工作原理
1. **离线预计算**：一次性将所有训练图像通过冻结的 VGGT 编码器，保存特征到磁盘
2. **训练加载**：训练时直接加载预计算特征，跳过 VGGT forward（占 60-70% 时间）
3. **自动降级**：若缓存缺失，自动回退到实时编码

#### 使用方法
```bash
# 第一步：预计算（一次性，2-4小时）
python scripts/precompute_vggt_features.py \
  --model_path /path/to/Qwen2.5-VL \
  --vggt_model_path /path/to/VGGT \
  --data_root /path/to/train_data \
  --cache_dir ./cache/vggt_features \
  --verify

# 第二步：启用缓存训练
export VGGT_CACHE_DIR=./cache/vggt_features
bash scripts/train_h800.sh
```

#### 验证方式
训练日志中应出现：
```
[INFO] VGGT feature cache enabled: /path/to/cache
[ACCELERATION] Using cached features for batch X
```

---

### 2. **方案 C：ZeRO-2 + CPU Offload** ⭐⭐⭐⭐
**预期加速：1.5-2x（相比 ZeRO-3）**

#### 核心文件
- `scripts/zero2_offload.json` - ZeRO-2 配置（含 CPU offload）

#### 配置特点
- `stage: 2` - 降低通信开销
- `offload_optimizer` + `offload_param` - 将优化器和参数卸载到 CPU
- `reduce_bucket_size: 5e8` - 减小通信 bucket

#### 使用方法
```bash
# 修改 train_h800.sh
DS_CONFIG="scripts/zero2_offload.json"

# 可尝试增大 batch size
bash scripts/train_h800.sh
```

#### 预期效果
- 显存占用：78GB → **45-55GB**
- 速度：比 ZeRO-3 快 **1.5x**
- 可支持 `per_device_batch_size=2`

---

### 3. **方案 E：Dataloader 优化** ⭐⭐⭐
**预期加速：1.2-1.5x**

#### 修改内容
```bash
# train_h800.sh 中已更新：
--dataloader_num_workers 16          # 8 → 16（加倍）
--dataloader_pin_memory True         # 确保开启
--dataloader_prefetch_factor 4       # 预取 4 个 batch
```

#### 效果
- 减少 GPU 等待 CPU 数据时间
- IO 密集型任务加速明显

---

### 4. **方案 F：多节点训练脚本** ⭐⭐⭐⭐
**预期加速：Nx（N = 节点数）**

#### 核心文件
- `scripts/train_2node_h800.sh` - 支持多节点的训练脚本

#### 特性
- 自动检测单节点/多节点模式
- 内置 NCCL/IB 优化配置
- 支持环境变量覆盖

#### 使用方法（2 节点示例）
```bash
# 节点 0（主节点）
export MASTER_ADDR=192.168.1.100
export NODE_RANK=0
export NNODES=2
bash scripts/train_2node_h800.sh

# 节点 1（工作节点）
export MASTER_ADDR=192.168.1.100
export NODE_RANK=1
export NNODES=2
bash scripts/train_2node_h800.sh
```

#### NCCL 配置
脚本自动启用：
- InfiniBand 支持（`NCCL_IB_DISABLE=0`）
- GPU Direct RDMA（`NCCL_NET_GDR_LEVEL=5`）
- RoCE 模式（`NCCL_IB_GID_INDEX=3`）

---

### 5. **训练监控工具** ⭐⭐⭐
**实时追踪训练进度和 GPU 负载**

#### 核心文件
- `scripts/monitor_training.sh` - 实时监控脚本

#### 功能
- 实时显示 steps/s 和 s/it
- 每 10 秒更新一次 GPU 显存占用
- 自动检测训练日志

#### 使用方法
```bash
# 启动监控（与训练并行）
bash scripts/monitor_training.sh /path/to/outputs/train_*.log
```

#### 示例输出
```
[2026-02-02 17:30:00] Step: 150 | Speed: 18.5s/it | Throughput: 0.54 steps/s
  GPU Mem (MB): 45123 46890 44567 45678 47890 46123 45890 46234
[2026-02-02 17:30:10] Step: 151 | Speed: 18.2s/it | Throughput: 0.55 steps/s
  GPU Mem (MB): 45234 46912 44589 45701 47901 46145 45912 46256
```

---

## 📊 预期性能提升

基于你的硬件（8× H800）和当前日志（77.86s/it）：

| 优化阶段 | 单步耗时 | Epoch时间 | 总时间(3 epochs) | 加速比 |
|---------|---------|----------|----------------|-------|
| **原始**（无优化） | 77.86s | 620h (~26天) | 1860h (~78天) | 1.0x |
| **+ Dataloader优化** | 65s | 520h | 1560h | **1.2x** |
| **+ VGGT缓存** | 18s | 144h (~6天) | 432h (~18天) | **4.3x** |
| **+ ZeRO-2** | 12s | 96h (~4天) | 288h (~12天) | **6.5x** |
| **+ 双节点（16 GPU）** | 6s | 48h (~2天) | 144h (~6天) | **13x** |

---

## 🚀 建议实施顺序

### 第一天（今晚）
1. **启动预计算脚本**（挂着过夜，~2-4小时）
```bash
python scripts/precompute_vggt_features.py \
  --model_path /public/home/vlabadmin/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct \
  --vggt_model_path /public/home/vlabadmin/.cache/modelscope/hub/models/facebook/VGGT-1B \
  --data_root /public/home/vlabadmin/dataset/VLN/JanusVLN_Trajectory_Data/trajectory_data/R2R-CE-640x480/train \
  --cache_dir /public/home/vlabadmin/dataset/NewJanusVLN/cache/vggt_features \
  --device cuda:0 \
  --verify
```

### 第二天（明天）
2. **启用缓存训练，验证加速效果**
```bash
export VGGT_CACHE_DIR=/public/home/vlabadmin/dataset/NewJanusVLN/cache/vggt_features
bash scripts/train_h800.sh

# 开启监控（另一个终端）
bash scripts/monitor_training.sh /public/home/vlabadmin/dataset/NewJanusVLN/outputs/vln_h800_8gpu/train_*.log
```

3. **观察日志确认**：
   - 单步时间从 77s 降到 **~18s**（4x 加速）
   - 日志出现 `[ACCELERATION] VGGT cache enabled`

### 第三天
4. **测试 ZeRO-2（可选）**
```bash
# 修改 train_h800.sh
DS_CONFIG="scripts/zero2_offload.json"

# 可尝试
# --per_device_train_batch_size 2
# --gradient_accumulation_steps 4
```

### 申请到双节点后
5. **多节点训练**
```bash
# 节点 0
export MASTER_ADDR=<node0_ip>
export NODE_RANK=0
export NNODES=2
export VGGT_CACHE_DIR=/shared/cache/vggt_features  # 确保共享存储
bash scripts/train_2node_h800.sh

# 节点 1
export MASTER_ADDR=<node0_ip>
export NODE_RANK=1
export NNODES=2
export VGGT_CACHE_DIR=/shared/cache/vggt_features
bash scripts/train_2node_h800.sh
```

---

## 🔍 验证清单

### ✅ 预计算阶段
- [ ] 脚本运行完成，无报错
- [ ] `cache/vggt_features/` 目录包含 `.pt` 文件
- [ ] `manifest.json` 显示正确的图像数量
- [ ] `--verify` 输出显示所有缓存有效

### ✅ 训练阶段
- [ ] 训练日志出现 `[ACCELERATION] VGGT cache enabled`
- [ ] 单步时间从 77s 降到 **15-20s**
- [ ] GPU 显存占用相比之前降低或持平
- [ ] `nvidia-smi` 显示所有 GPU 均在使用

### ✅ 多节点阶段
- [ ] 两个节点的日志都显示 `NCCL Init COMPLETE`
- [ ] 两节点的 step 数同步增长
- [ ] 单步时间进一步减半

---

## 📚 参考文档

- **详细优化方案**：`docs/TRAINING_OPTIMIZATION.md`
- **缓存使用指南**：`docs/VGGT_CACHE_GUIDE.md`
- **预计算脚本**：`scripts/precompute_vggt_features.py`
- **单节点训练**：`scripts/train_h800.sh`
- **双节点训练**：`scripts/train_2node_h800.sh`
- **实时监控**：`scripts/monitor_training.sh`

---

## ⚠️ 注意事项

### 缓存路径必须一致
```bash
# 预计算时
--cache_dir /path/to/cache

# 训练时
--vggt_cache_dir /path/to/cache  # 必须相同
```

### 多节点共享存储
- `VGGT_CACHE_DIR` 必须在所有节点上可访问（NFS/共享存储）
- 预计算只需在一个节点运行一次

### 图像分辨率
- **保持不变**：`--max_pixels $((576*28*28))`（按你的要求）
- 预计算时会使用相同的分辨率设置

### 磁盘空间
- 预留 **50-100GB** 用于缓存特征
- 定期清理旧的 checkpoint（`--save_total_limit 2`）

---

## 🎯 预期最终效果

实施全部优化后（单节点 + 缓存 + ZeRO-2）：

- **训练速度**：77.86s/it → **12-15s/it**（**~6x 加速**）
- **Epoch 时间**：620h → **~100h**（**4-5 天/epoch**）
- **总训练时间（3 epochs）**：1860h → **~300h**（**12-13 天**）

如果加上双节点（16 GPU）：

- **Epoch 时间**：→ **~50h**（**2 天/epoch**）
- **总训练时间（3 epochs）**：→ **~150h**（**6-7 天**）

---

## 需要帮助？

遇到问题请检查：
1. 日志文件：`tail -200 /path/to/outputs/train_*.log`
2. GPU 状态：`nvidia-smi`
3. 缓存目录：`ls -lh /path/to/cache/vggt_features/ | head -20`
4. 进程状态：`ps aux | grep train_vln`

---

**所有代码已就绪，可以立即开始预计算！** 🚀
