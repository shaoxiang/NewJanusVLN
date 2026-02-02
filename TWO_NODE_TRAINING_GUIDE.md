# 两节点 IB 训练配置指南

## ✅ 已完成修改

### 1. 移除 VGGT 缓存相关代码
所有 VGGT 缓存功能已安全移除，模型将实时计算 VGGT features。

**修改文件：**
- `src/qwen_vl/train/argument.py` - 移除 `use_vggt_cache` 参数
- `src/qwen_vl/data/vln_data.py` - 移除缓存加载和处理逻辑
- `src/qwen_vl/model/modeling_qwen2_5_vl.py` - 移除缓存使用分支
- `scripts/train_2node_h800.sh` - 移除缓存相关配置

### 2. 两节点训练优化

#### NCCL 优化配置
针对 InfiniBand 和 H800 GPU 的高级优化：

```bash
# InfiniBand 基础配置
export NCCL_IB_DISABLE=0              # 启用 IB
export NCCL_IB_HCA=mlx5               # IB 设备
export NCCL_SOCKET_IFNAME=ib0         # 网络接口
export NCCL_NET_GDR_LEVEL=5           # GPU Direct RDMA
export NCCL_IB_GID_INDEX=3            # RoCE 模式

# H800 专用优化
export NCCL_IB_TIMEOUT=22             # IB 超时（增加稳定性）
export NCCL_IB_RETRY_CNT=7            # IB 重试次数
export NCCL_CROSS_NIC=0               # 禁用跨 NIC（提高稳定性）
export NCCL_P2P_LEVEL=SYS             # P2P 级别
export NCCL_BUFFSIZE=8388608          # 缓冲区大小 8MB
export NCCL_NTHREADS=640              # H800 NCCL 线程数
```

#### DeepSpeed ZeRO-3 优化
`scripts/zero3.json` 已优化：

```json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,              // 计算与通信重叠
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,         // 500MB（适合 IB）
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9
  }
}
```

#### 训练脚本优化
`scripts/train_2node_h800.sh` 新增：

```bash
export OMP_NUM_THREADS=8              # OpenMP 线程数
export CUDA_LAUNCH_BLOCKING=0         # 异步 CUDA 启动
--ddp_timeout 7200                    # DDP 超时 2 小时
--ddp_find_unused_parameters False    # 禁用未使用参数查找（加速）
```

---

## 🚀 使用方法

### 单节点训练（8 GPUs）
```bash
bash scripts/train_2node_h800.sh
```

### 两节点训练（16 GPUs）

**Node 0 (Master):**
```bash
export MASTER_ADDR=192.168.1.100  # 替换为你的 Master IP
export NODE_RANK=0
export NNODES=2
bash scripts/train_2node_h800.sh
```

**Node 1 (Worker):**
```bash
export MASTER_ADDR=192.168.1.100  # 与 Master 相同的 IP
export NODE_RANK=1
export NNODES=2
bash scripts/train_2node_h800.sh
```

### 关键环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NNODES` | 1 | 总节点数 |
| `NODE_RANK` | 0 | 当前节点编号（0 为 master） |
| `MASTER_ADDR` | localhost | Master 节点 IP 地址 |
| `MASTER_PORT` | 29500 | Master 端口 |
| `NPROC_PER_NODE` | 8 | 每节点 GPU 数 |

---

## 🔧 硬件特定配置

### 检查 InfiniBand 设备
```bash
# 查看可用 IB 设备
ibv_devices

# 查看 IB 状态
ibstat

# 查看网络接口
ifconfig | grep ib
```

### 调整 NCCL 配置
根据你的硬件调整以下变量：

1. **如果没有 InfiniBand（使用以太网）：**
   ```bash
   export NCCL_IB_DISABLE=1
   export NCCL_SOCKET_IFNAME=eth0  # 或你的以太网接口
   ```

2. **如果 IB 设备名不是 `mlx5`：**
   ```bash
   ibv_devices  # 查看设备名
   export NCCL_IB_HCA=<your_device_name>
   ```

3. **如果网络接口名不是 `ib0`：**
   ```bash
   ifconfig | grep ib  # 查看接口名
   export NCCL_SOCKET_IFNAME=<your_ib_interface>
   ```

---

## 📊 性能监控

### 训练期间监控
```bash
# 实时查看 GPU 使用率
watch -n 1 nvidia-smi

# 查看 IB 网络流量
watch -n 1 "ibstat | grep -A 5 'Port 1'"

# 查看进程网络连接
netstat -anp | grep <master_port>
```

### TensorBoard
```bash
tensorboard --logdir outputs/vln_2node_h800 --port 6006
```

---

## ⚠️ 常见问题排查

### 1. NCCL 初始化失败
**症状：** `NCCL_ERROR` 或超时
**解决：**
```bash
# 检查节点间网络连通性
ping <other_node_ip>

# 检查端口是否可访问
telnet <master_addr> 29500

# 增加 NCCL 超时
export NCCL_IB_TIMEOUT=30
export NCCL_ASYNC_ERROR_HANDLING=1
```

### 2. OOM (Out of Memory)
**解决：**
```bash
# 减小 batch size
export PER_DEVICE_BATCH=1
export GRAD_ACCUM_STEPS=16

# 或减少历史帧数
export MAX_HISTORY_IMAGES=4
```

### 3. 训练速度慢
**检查：**
- IB 是否真正启用：`export NCCL_DEBUG=INFO` 查看日志中的 `Using network IB`
- 数据加载是否瓶颈：减少 `dataloader_num_workers` 或增加 `prefetch_factor`
- 是否有跨节点通信开销：检查 `overlap_comm=true` 是否生效

### 4. 节点间不同步
**解决：**
```bash
# 确保所有节点代码一致
git status  # 在所有节点上检查

# 确保所有节点使用相同的随机种子和数据顺序
# 在训练脚本中已默认处理
```

---

## 📈 预期性能

### 有效 Batch Size
```
Effective Batch Size = per_device_batch × num_gpus × num_nodes × grad_accum_steps
                     = 1 × 8 × 2 × 8
                     = 128
```

### 训练速度估算
- **单节点 (8×H800):** ~X steps/sec
- **双节点 (16×H800):** ~1.7-1.9X steps/sec（理论 2X，实际受通信开销影响）

### GPU 利用率目标
- **单节点:** 95%+
- **双节点:** 85-95%（通信开销导致略低）

---

## 🎯 生产环境建议

### 训练稳定性
```bash
# 禁用调试信息（生产环境）
# 在 scripts/train_2node_h800.sh 中注释掉：
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 启用自动重启（可选）
--max_retries 3
```

### 检查点策略
```bash
# 当前配置
--save_steps 500
--save_total_limit 2

# 建议：根据训练时长调整
# 如果训练 > 24 小时，考虑更频繁保存
--save_steps 200
--save_total_limit 3
```

### 日志管理
```bash
# 日志文件：outputs/vln_2node_h800/train_node0_*.log
# 日志文件：outputs/vln_2node_h800/train_node1_*.log

# 定期清理旧日志
find outputs/vln_2node_h800 -name "*.log" -mtime +7 -delete
```

---

## 📝 修改总结

### 删除的功能
- ❌ VGGT features 预计算缓存
- ❌ 缓存文件加载逻辑（`.vggt_cache.pt`）
- ❌ `--use_vggt_cache` 命令行参数

### 新增的功能
- ✅ H800 专用 NCCL 优化配置
- ✅ IB 网络高级调优
- ✅ DeepSpeed ZeRO-3 通信优化
- ✅ DDP 超时和参数优化
- ✅ OpenMP 和 CUDA 环境配置

### 保持不变
- ✅ VGGT 实时前向计算（上下文相关）
- ✅ 所有模型架构和训练逻辑
- ✅ 数据加载和预处理流程
- ✅ 损失函数和优化器配置

---

如有任何问题或需要进一步优化，请参考日志文件或联系技术支持。
