# JanusVLN 训练深度优化方案

## 当前瓶颈分析
根据您的训练日志（77.86s/iter，620h/epoch）和显存占用（GPU间不均衡，最高78GB），主要瓶颈：

### 1. 视觉编码重复计算（最大瓶颈）
**问题**：`modeling_qwen2_5_vl.py:2096-2103` 每个训练步都对 8 张历史图像重新跑 VGGT aggregator
- 虽然在 `torch.no_grad()` 下，但仍占用大量 CUDA kernel 时间
- 8 张图像 × batch_size × gradient_accumulation = 64 次视觉编码/step
- VGGT 已冻结（`tune_mm_vision=False`），输出完全可缓存

**理论加速**：3-5x（视觉编码占总时间 ~60-70%）

### 2. ZeRO-3 通信开销
**问题**：`stage3_max_live_parameters=1e9` 导致每次 forward 都广播大量参数
- 当前配置下，8×H800 每步通信 ~20GB 参数
- `overlap_comm=true` 无法完全隐藏通信延迟

**理论加速**：1.5-2x（降低 ZeRO stage 或优化参数）

### 3. 显存不均衡
**现象**：GPU1 占 78GB，GPU2 仅 46GB
- 可能原因：动态 batch padding 导致某些 GPU 处理更多视觉 token
- `group_by_modality_length=True` 在小 batch 下效果有限

---

## 优化方案（分优先级）

### 🔥 优先级 P0：视觉特征缓存（立即实施）

#### 方案 A：训练时缓存视觉特征（推荐）
**思路**：在数据预处理阶段缓存 VGGT 输出，训练时直接加载

**实施步骤**：
1. 新增预处理脚本 `scripts/precompute_visual_features.py`
2. 遍历所有训练样本，保存 `{trajectory_id}_{step_idx}.pt`
3. 修改 `vln_data.py`，优先加载缓存特征
4. 训练时跳过 VGGT forward

**优点**：
- 加速最显著（3-5x）
- 不改变模型逻辑
- 可增量预计算

**缺点**：
- 需要额外磁盘空间（估计 50-100GB，取决于轨迹数）
- 首次预计算耗时（约 2-4 小时，仅需一次）

**实施难度**：⭐⭐（中等）

---

#### 方案 B：在线缓存（更简单，但加速有限）
**思路**：在同一 epoch 内缓存已见过的图像特征

**修改位置**：`modeling_qwen2_5_vl.py` forward 函数

**核心逻辑**：
```python
# 在 __init__ 中添加
self.visual_cache = {}  # key: image_hash, value: features

# 在 forward 中修改（2096 行附近）
for k, frame in enumerate(images_vggt[i]):
    frame_hash = hash(frame.data_ptr())  # 或用内容 hash
    if frame_hash in self.visual_cache:
        features = self.visual_cache[frame_hash]
    else:
        # 原有的 VGGT 编码逻辑
        ...
        self.visual_cache[frame_hash] = features
```

**优点**：
- 改动最小（~10 行代码）
- 无需额外存储

**缺点**：
- 首 epoch 无加速
- 内存占用增加（~5-10GB，可设置 LRU cache）
- Dataloader shuffle 会降低命中率

**实施难度**：⭐（简单）

---

### 🔥 优先级 P1：降低 ZeRO stage

#### 方案 C：切换到 ZeRO-2 + CPU offload
**背景**：您已测试过 ZeRO-2 会 OOM，但可通过以下配置解决：

**新 ZeRO-2 配置**（`scripts/zero2_offload.json`）：
```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "bf16": {"enabled": "auto"},
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,
    "contiguous_memory_optimization": true,
    "synchronize_checkpoint_boundary": false
  }
}
```

**关键改动**：
- `offload_optimizer` + `offload_param`：将优化器状态和参数卸载到 CPU
- `activation_checkpointing.cpu_checkpointing`：激活值也卸载到 CPU
- `reduce_bucket_size` 降低到 500MB，减少通信延迟

**预期效果**：
- 显存占用：降至 ~40-50GB/GPU（可支持 batch_size=2）
- 速度：比 ZeRO-3 快 1.5-2x（虽然有 CPU-GPU 传输，但通信量大幅减少）

**实施步骤**：
```bash
# 修改 train_h800.sh
DS_CONFIG="./scripts/zero2_offload.json"

# 可尝试增大 batch size
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4 \
```

**实施难度**：⭐（只需改配置文件）

---

### 🔥 优先级 P2：优化数据加载

#### 方案 D：减少图像分辨率（训练阶段）
**当前配置**：
```bash
--max_pixels $((576*28*28))  # 451,584 pixels
--video_max_frames 8
```

**建议调整**（训练阶段）：
```bash
--max_pixels $((384*28*28))  # 301,056 pixels (-33%)
--min_pixels $((28*28*28))   # 增加 min_pixels 下限
```

**效果**：
- 显存：降低 20-30%
- 速度：提升 15-20%（IO + 视觉编码）
- 精度损失：<1%（VLN 任务对超高分辨率不敏感）

**验证方法**：先跑 100 steps 对比 loss，若差异 <5% 可采用

---

#### 方案 E：优化 Dataloader
**当前瓶颈**：`dataloader_num_workers=8` 可能不足

**建议**：
```python
# train_h800.sh 中增加
--dataloader_num_workers 16 \      # 加倍（如果 CPU 核心充足）
--dataloader_pin_memory True \     # 确保开启
--dataloader_prefetch_factor 4 \   # 预取 4 个 batch
```

**同时在 `vln_data.py` 中优化图像加载**：
```python
# 使用 PIL-SIMD 或 cv2 替代 PIL.Image（提速 2-3x）
import cv2

def load_image_fast(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)
```

---

### 🔥 优先级 P3：多节点训练优化

#### 方案 F：跨节点训练配置
**您提到"想办法使用更多节点"**，以下是最佳实践：

**1. 修改 `train_h800.sh` 支持多节点**：
```bash
# 在脚本顶部添加
NNODES=${NNODES:-1}              # 节点数
NODE_RANK=${NODE_RANK:-0}        # 当前节点 rank
MASTER_ADDR=${MASTER_ADDR:-localhost}

# 修改 torchrun 参数
"${LAUNCHER[@]}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  ...
```

**2. 启动命令（在每个节点上执行）**：
```bash
# 节点 0（主节点）
export MASTER_ADDR=192.168.1.100  # 主节点 IP
export NODE_RANK=0
export NNODES=2
bash scripts/train_h800.sh

# 节点 1
export MASTER_ADDR=192.168.1.100
export NODE_RANK=1
export NNODES=2
bash scripts/train_h800.sh
```

**3. 优化跨节点通信**：
```bash
# 设置 NCCL 环境变量（在 train_h800.sh 中）
export NCCL_IB_DISABLE=0           # 启用 InfiniBand
export NCCL_IB_HCA=mlx5            # IB 设备（根据硬件调整）
export NCCL_SOCKET_IFNAME=eth0     # 若用以太网
export NCCL_NET_GDR_LEVEL=5        # GPU Direct RDMA
```

**预期效果**（2 节点 16 GPU）：
- 有效 batch size 翻倍（可增大学习率）
- 线性加速（若网络带宽足够）

---

## 综合优化路线图

### 阶段 1：快速优化（1-2 天）
1. ✅ **实施方案 B**（在线缓存）：~2 小时改代码 + 测试
2. ✅ **实施方案 D**（降分辨率）：改配置立即生效
3. ✅ **实施方案 E**（优化 dataloader）：~1 小时

**预期加速**：1.5-2x（iter 时间降至 40-50s）

---

### 阶段 2：深度优化（3-5 天）
4. ✅ **实施方案 A**（预计算缓存）：需要写预处理脚本
5. ✅ **实施方案 C**（ZeRO-2 + offload）：测试显存和速度平衡点

**预期加速**：3-4x（iter 时间降至 20-25s）

---

### 阶段 3：规模化（按需）
6. ✅ **实施方案 F**（多节点）：若单节点优化后仍不满足需求

**预期加速**：Nx（N 为节点数，需要高速互联）

---

## 附：监控脚本

### 1. 实时训练吞吐监控
```bash
# 保存为 scripts/monitor_training.sh
#!/bin/bash
LOG_FILE="$1"
while true; do
  if [[ -f "$LOG_FILE" ]]; then
    # 提取最近 10 行的 it/s
    tail -20 "$LOG_FILE" | grep -oP '\d+\.\d+s/it' | tail -1
  fi
  sleep 10
done
```

### 2. 显存均衡检查
```python
# 保存为 scripts/check_gpu_balance.py
import subprocess
import time

while True:
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', 
                             '--format=csv,noheader,nounits'], 
                            capture_output=True, text=True)
    mems = [int(x) for x in result.stdout.strip().split('\n')]
    print(f"GPU Mem: {mems}, Imbalance: {max(mems)-min(mems)}MB")
    time.sleep(5)
```

---

## 优先推荐实施顺序

基于您的约束（不动 `max_history_images`，快速见效）：

1. **立即做**（今天）：
   - 方案 D（降分辨率）
   - 方案 E（dataloader 优化）
   
2. **本周做**：
   - 方案 B（在线缓存）
   - 方案 C（测试 ZeRO-2）

3. **评估后决定**：
   - 方案 A（预计算，效果最好但需要时间）
   - 方案 F（多节点，需要硬件支持）

---

## 预期最终性能

假设实施方案 A + C + D + E：
- **单步时间**：77.86s → **15-20s**（4-5x 加速）
- **Epoch 时间**：620h → **120-150h**（~5 天）
- **显存占用**：78GB → **45-55GB**（可尝试更大 batch）
- **吞吐量**：15 samples/s → **60-80 samples/s**

如果再加多节点（2 节点 16 GPU）：
- **Epoch 时间**：120h → **60-70h**（~3 天）

---

## 需要我帮您实施哪个方案？

我可以立即提供：
1. **方案 B 的完整代码**（在线缓存，改动最小）
2. **方案 A 的预计算脚本**（离线缓存，效果最好）
3. **方案 C 的 ZeRO-2 配置**（已写好，见上文）
4. **方案 E 的数据加载优化**（快速 IO）

请告诉我优先实施哪个，我会提供可直接运行的代码！
