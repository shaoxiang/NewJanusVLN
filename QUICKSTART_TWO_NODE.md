# 快速启动指南 - 两节点 IB 训练

## ✅ 已完成的修改

### 1. **移除所有 VGGT 缓存相关代码**
   - ✅ 删除 `use_vggt_cache` 参数和逻辑
   - ✅ 删除缓存加载功能
   - ✅ 删除缓存文件处理
   - ✅ **保留 VGGT 实时计算（上下文相关的序列建模）**

### 2. **优化两节点 IB 训练配置**
   - ✅ NCCL 针对 H800 + InfiniBand 优化
   - ✅ DeepSpeed ZeRO-3 通信参数优化
   - ✅ DDP 超时和参数调优
   - ✅ 数据加载和内存管理优化

---

## 🚀 立即开始训练

### 步骤 1: 设置环境变量

**Node 0 (Master 节点):**
```bash
export MODEL_PATH="/path/to/Qwen2.5-VL-3B-Instruct"
export VGGT_MODEL_PATH="/path/to/VGGT-1B"
export DATA_ROOT="/path/to/train_data"

export MASTER_ADDR=192.168.1.100  # 替换为你的 Master IP
export NODE_RANK=0
export NNODES=2
export MASTER_PORT=29500
```

**Node 1 (Worker 节点):**
```bash
export MODEL_PATH="/path/to/Qwen2.5-VL-3B-Instruct"
export VGGT_MODEL_PATH="/path/to/VGGT-1B"
export DATA_ROOT="/path/to/train_data"

export MASTER_ADDR=192.168.1.100  # 与 Master 相同
export NODE_RANK=1
export NNODES=2
export MASTER_PORT=29500
```

### 步骤 2: 检查配置（可选但推荐）
```bash
bash scripts/check_multi_node.sh
```

### 步骤 3: 启动训练

**在两个节点上同时运行：**
```bash
bash scripts/train_2node_h800.sh
```

---

## ⚙️ 关键配置说明

### NCCL 配置（已在脚本中自动设置）
```bash
# InfiniBand 配置
export NCCL_IB_DISABLE=0              # 启用 IB
export NCCL_IB_HCA=mlx5               # IB 设备
export NCCL_SOCKET_IFNAME=ib0         # IB 网络接口
export NCCL_IB_TIMEOUT=22             # IB 超时
export NCCL_BUFFSIZE=8388608          # 8MB 缓冲区

# H800 优化
export NCCL_NTHREADS=640              # NCCL 线程数
export NCCL_NET_GDR_LEVEL=5           # GPU Direct RDMA
```

### DeepSpeed ZeRO-3（已优化）
- **reduce_bucket_size:** 500MB（适合 IB 高带宽）
- **overlap_comm:** true（计算与通信重叠）
- **stage3_prefetch_bucket_size:** 500MB

### 训练参数（可调整）
```bash
export PER_DEVICE_BATCH=1             # 每 GPU batch size
export GRAD_ACCUM_STEPS=8             # 梯度累积步数
export MAX_HISTORY_IMAGES=8           # 历史帧数
export LEARNING_RATE=2e-5             # 学习率
```

**有效 Batch Size = 1 × 8 GPUs × 2 nodes × 8 accum = 128**

---

## 🔍 监控和调试

### 实时监控
```bash
# GPU 使用率
watch -n 1 nvidia-smi

# IB 网络流量
watch -n 1 "ibstat | grep -A 5 'Port 1'"

# 训练日志
tail -f outputs/vln_2node_h800/train_node0_*.log
```

### TensorBoard
```bash
tensorboard --logdir outputs/vln_2node_h800 --port 6006
```

---

## ⚠️ 常见问题

### Q1: 如果没有 InfiniBand，只有以太网？
**A:** 修改 NCCL 配置：
```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0  # 或你的网络接口名
```

### Q2: NCCL 初始化超时？
**A:** 检查网络连通性和防火墙：
```bash
# 测试连通性
ping <other_node_ip>
telnet <master_addr> 29500

# 增加超时
export NCCL_IB_TIMEOUT=30
```

### Q3: OOM (显存不足)？
**A:** 减小 batch size 或历史帧数：
```bash
export PER_DEVICE_BATCH=1
export GRAD_ACCUM_STEPS=16
export MAX_HISTORY_IMAGES=4
```

### Q4: 训练速度慢？
**A:** 检查：
- IB 是否启用：查看日志中的 `Using network IB`
- GPU 利用率：应该 > 85%
- 数据加载瓶颈：尝试调整 `dataloader_num_workers`

---

## 📊 预期性能

| 配置 | 步/秒 | GPU 利用率 |
|------|-------|------------|
| 单节点 (8×H800) | ~X | 95%+ |
| 双节点 (16×H800) | ~1.7-1.9X | 85-95% |

---

## 📁 相关文件

- **训练脚本:** `scripts/train_2node_h800.sh`
- **DeepSpeed 配置:** `scripts/zero3.json`
- **配置检查:** `scripts/check_multi_node.sh`
- **详细文档:** `TWO_NODE_TRAINING_GUIDE.md`

---

## 📝 修改清单

| 文件 | 修改内容 |
|------|----------|
| `src/qwen_vl/train/argument.py` | 删除 `use_vggt_cache` 参数 |
| `src/qwen_vl/data/vln_data.py` | 删除缓存加载逻辑 |
| `src/qwen_vl/model/modeling_qwen2_5_vl.py` | 删除缓存使用分支 |
| `scripts/train_2node_h800.sh` | 优化 NCCL 和 DDP 配置 |
| `scripts/zero3.json` | 优化 ZeRO-3 通信参数 |

---

**所有修改已完成，可以直接开始两节点训练！** 🎉

如有问题，请查看日志文件或参考 `TWO_NODE_TRAINING_GUIDE.md` 详细文档。
