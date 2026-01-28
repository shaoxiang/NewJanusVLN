---
name: newjanusvln-speed-and-model-review
overview: 梳理 NewJanusVLN 的模型/输入输出/设计风险，并基于 H800 8卡训练日志制定提速与显存利用优化方案。
todos:
  - id: repo-scan
    content: 使用[subagent:code-explorer]扫描训练入口与关键调用链
    status: completed
  - id: model-io-review
    content: 输出模型/数据输入输出与创新点结构化梳理
    status: completed
    dependencies:
      - repo-scan
  - id: risk-assessment
    content: 整理设计风险清单与可验证检查项
    status: completed
    dependencies:
      - model-io-review
  - id: log-bottleneck-analysis
    content: 基于日志定位瓶颈并给出证据与优先级
    status: completed
    dependencies:
      - repo-scan
  - id: speedup-experiment-matrix
    content: 制定8卡H800提速实验矩阵与预期收益表
    status: completed
    dependencies:
      - log-bottleneck-analysis
  - id: memory-batch-plan
    content: 给出增大batch与显存利用上限建议
    status: completed
    dependencies:
      - speedup-experiment-matrix
  - id: model-optimization-roadmap
    content: 输出模型层优化路线与回退策略
    status: completed
    dependencies:
      - risk-assessment
      - log-bottleneck-analysis
---

## Product Overview

输出一份针对 NewJanusVLN 的“模型与数据流复盘 + 风险评估 + 单机8卡提速方案”报告，并给出可直接照做的实验清单与参数建议。

## Core Features

- **模型/输入输出梳理**：以结构化表格描述训练样本字段、图像/历史帧组织方式、文本模板分段、模型前向关键分支与最终监督信号的对应关系。
- **创新点与设计风险评估**：总结 VGGT 特征注入与分段加权损失的作用点，列出可能导致效果不稳/训练变慢/显存浪费的风险项，并给出可验证的检查方法。
- **训练日志诊断**：从现有训练日志中提取吞吐、保存抖动、潜在等待点等证据，形成“问题—证据—影响—优先级”的排查结论。
- **提速与显存利用优化方案**：给出单节点 8×H800 下尽量不掉效果的提速路径（含是否能增大 batch、如何改累积、如何减少保存开销等），并提供分阶段试验建议与预期收益对比表。
- **模型层面优化思路**：给出不改任务定义前提下的模型侧加速/降显存路线（历史帧策略、注入方式、序列长度控制等），并给出风险与回退策略。

## Tech Stack（基于现有工程）

- Python + PyTorch（分布式：torchrun）
- HuggingFace Transformers（Trainer/TrainingArguments）
- DeepSpeed ZeRO-3（`scripts/zero3.json`）
- bf16 + FlashAttention2
- 日志与实验：TensorBoard（脚本当前配置），可选 WandB（仓库依赖已包含）

## Current Architecture Snapshot（关键链路）

- 训练入口：`scripts/train_h800.sh` → `qwen_vl.train.train_vln`
- 数据集/拼batch：`src/qwen_vl/data/vln_data.py`
- 每样本多张历史图 + 当前图；逐张做图像处理，并组装为 `pixel_values / image_grid_thw / images_vggt`
- 分段构造 `loss_weights / segment_ids`，Trainer 内分段统计 loss
- 模型：`src/qwen_vl/model/modeling_qwen2_5_vl.py`
- Qwen2.5-VL 视觉特征 + VGGT 3D特征（`image_embeds += lam * image_embeds_3d`）
- 训练器：`src/qwen_vl/train/train_vln.py`（`WeightedLossTrainer`）

## Evidence from Log（已定位的关键现象）

- `Num examples=612,492`，`Total train batch size=64`，`Total optimization steps=28,710`
- 训练约 `~55s/it`（优化 step 粒度），且每 `save_steps=500` 会触发明显额外变慢（ZeRO-3 gather 权重保存）
- 用户侧观测 GPU util 约 9–12% 且显存有余量：高度怀疑 **输入管线/保存/同步等待** 为主要瓶颈（需进一步用 profile 验证）

## Data Flow Diagram

```mermaid
flowchart LR
  A[Episode目录/PNG+JSON] --> B[LazySupervisedDataset.__getitem__]
  B --> C[图像处理: load_and_preprocess_images + image_processor]
  C --> D[组装: pixel_values/image_grid_thw/images_vggt]
  D --> E[DataCollator 拼接与pad: input_ids/labels/position_ids]
  E --> F[WeightedLossTrainer]
  F --> G[Model forward: Qwen2.5-VL visual + VGGT + 注入]
  G --> H[loss(分段加权) + backward]
  H --> I[DeepSpeed ZeRO-3 step]
  I --> J[checkpoint save(gather 16bit weights)]
```

## Implementation Focus（提速与显存优化的技术抓手，按优先级）

1. **数据加载与图像预处理加速**（最可能的主瓶颈）

- 减少 `__getitem__` 中重复开销（如频繁 deepcopy、逐张处理、插值/对齐）
- 提升 DataLoader 并行与稳定吞吐（persistent workers、prefetch、pin memory、workers 数量与CPU亲和）
- 中期方案：将“图像→模型所需张量”离线缓存（按路径/分辨率/patch对齐参数做key），训练时直接读缓存

2. **减少 checkpoint 抖动**

- 评估将 `save_steps` 放大、只保留关键里程碑、或关闭 ZeRO-3 保存时 gather（改为训练后再合并权重）

3. **用“更大 micro-batch + 更小累积”换吞吐**

- 在显存允许下，尝试 `per_device_train_batch_size ↑` 且 `gradient_accumulation_steps ↓` 保持总batch不变，以减少一次优化 step 的 forward/backward 次数

4. **训练计算侧加速/降显存**

- 若显存确有余量：评估减小/关闭部分 gradient checkpointing（以速度换显存）
- 优化 VGGT 分支：减少帧数、复用缓存、或将 3D 特征预计算落盘
- 约束有效序列长度（避免超大 `model_max_length` 带来的 padding/对齐成本），并用统计验证真实 token 分布

## Modified / New Files（预计会触达的代码点）

```text
w:/workspace/VLA & VLN/NewJanusVLN/
├── scripts/train_h800.sh                 # 参数试验矩阵落点（batch/accum/save等）
├── scripts/zero3.json                    # checkpoint/gather策略评估与调整
├── src/qwen_vl/data/vln_data.py          # 数据吞吐优化/缓存策略/预处理优化
├── src/qwen_vl/train/train_vln.py        # 增加profile钩子/统计数据耗时
└── src/qwen_vl/model/modeling_qwen2_5_vl.py # VGGT分支耗时与缓存/帧策略优化
```

## Agent Extensions

### SubAgent

- **code-explorer**
- Purpose: 快速跨目录检索训练/数据/模型/日志相关实现与可疑瓶颈点
- Expected outcome: 形成“关键函数位置 + 调用链 + 可改动点”的清单，支撑后续优化与实验设计