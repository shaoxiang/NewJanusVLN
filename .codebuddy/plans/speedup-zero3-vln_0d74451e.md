---
name: speedup-zero3-vln
overview: 在必须使用 DeepSpeed ZeRO-3 且不降低视觉 token 的前提下，针对 allocator cache flush（高显存压力）与 step time 过长，调整 ZeRO-3 配置与 CUDA allocator 行为并增加可观测性，优先提升稳定性与吞吐。
todos:
  - id: baseline-and-bottleneck-summary
    content: 用现有日志与DS输出定义性能基线与主要瓶颈指标
    status: completed
  - id: tune-zero3-config
    content: 修改 scripts/zero3.json：显式bucket/prefetch/persistence并开启wall_clock_breakdown
    status: completed
    dependencies:
      - baseline-and-bottleneck-summary
  - id: allocator-env-tuning
    content: 修改 scripts/train_h800.sh：加入PYTORCH_CUDA_ALLOC_CONF等稳健内存环境变量
    status: completed
    dependencies:
      - baseline-and-bottleneck-summary
  - id: fix-metrics-and-cache-callback
    content: 修改 train_vln.py/argument.py：修正图片数日志并加入可选同步empty_cache回调
    status: completed
    dependencies:
      - baseline-and-bottleneck-summary
  - id: repo-wide-verify-with-subagent
    content: 使用 [subagent:code-explorer] 复核相似实现与潜在冲突点，确保可回退
    status: completed
    dependencies:
      - tune-zero3-config
      - fix-metrics-and-cache-callback
  - id: validation-checklist
    content: 制定验证清单：flush频率、step_time稳定性、DS分解耗时与显存曲线
    status: completed
    dependencies:
      - allocator-env-tuning
      - repo-wide-verify-with-subagent
---

## User Requirements

- 必须继续使用 DeepSpeed ZeRO-3（否则 OOM），保持 `max_history_images=8` 与现有数据规模不变。
- 在不降低视觉 token 的前提下，显著缩短训练单 step 时间或降低波动，并优先消除/减少 `allocator cache flush` 警告造成的性能损失。
- 增强可诊断性：能从日志中更准确区分「图片张数」与「视觉 token/patch 数」，并获得更清晰的耗时/通信分解信息。

## Product Overview

- 现有 VLN 训练流程的性能稳定性与吞吐优化：通过 ZeRO-3 配置、CUDA allocator 配置与轻量训练回调，减少显存高压导致的缓存抖动与额外同步开销，同时输出可用于定位瓶颈的关键性能指标。

## Core Features

- ZeRO-3 配置稳健调参（bucket/prefetch/persistence 等）以降低显存峰值与碎片化风险，并开启 wall-clock breakdown。
- 启动脚本增加 PyTorch CUDA allocator 相关环境变量，减少频繁 cache flush。
- 训练日志指标修正：输出真实 `num_images_per_sample`（≈9）与 `visual_tokens_per_sample`（merge 前/后口径可选），避免将 patch/token 数误记为图片数。
- 可控的跨 rank 同步 `empty_cache`（可开关、低频触发）以在高压阶段避免不一致 flush 抖动。

## Tech Stack Selection（沿用现有项目）

- 训练框架：PyTorch + HuggingFace Transformers Trainer
- 分布式与并行：torch.distributed + NCCL + DeepSpeed ZeRO-3
- 混精：bf16
- 监控：Trainer Callback 日志 + DeepSpeed wall_clock_breakdown

## Implementation Approach

- 策略：不改数据/视觉 token（按用户选择），先从「显存高压导致的 allocator cache flush」入手，通过 **(1) CUDA allocator 策略** + **(2) ZeRO-3 bucket/prefetch/persistence 参数显式化** + **(3) 低频、跨 rank 同步 empty_cache** 来降低碎片化与不一致 flush；同时开启 DeepSpeed wall-clock breakdown 以获得更可操作的耗时分解。
- 关键决策与权衡：
- bucket/prefetch 显式设定：更稳健但需小范围试探（过小增加通信，过大增加峰值内存）。选择“稳健优先”的中间档，并保留回退为 auto 的能力。
- empty_cache：默认关闭或低频触发；只在观察到 flush 高频/显存紧张时启用，避免引入额外同步开销。
- 性能/可靠性：
- 主要瓶颈预期在：LLM 长序列注意力（不可改 token 前提下）、ZeRO-3 参数 gather/通信、以及 allocator flush 引发的同步与重分配。
- 目标是减少 flush 次数与波动，使 step_time 更接近稳定下限；通过 wall_clock_breakdown 定位通信与计算占比，避免盲调。

## Implementation Notes（基于已探索代码）

- `train_vln.py` 当前把 `pixel_values.shape[0]` 当作 `images_per_sample`，但在 VLN collator 中 `pixel_values` 第一维并非“图片张数”口径；应改用 `image_grid_thw.shape[0] / batch_size` 输出真实图片数（≈9），并将 `visual_tokens_per_sample` 明确标注为 merge 前 token（或同时输出 merge 后估计）。
- DeepSpeed 配置 `scripts/zero3.json` 目前 bucket/prefetch/persistence 多为 `auto`，且未开启 wall-clock breakdown；这不利于稳健排查与定位。
- 训练脚本 `scripts/train_h800.sh` 适合加入 `PYTORCH_CUDA_ALLOC_CONF` 等环境变量以降低碎片化与 cache flush。

## Architecture Design

- 不改整体训练架构；在现有 `train_vln.py` 的 Trainer Callback 体系内新增/扩展轻量回调；在 `scripts/zero3.json` 与 `scripts/train_h800.sh` 做配置层优化。
- 数据流不变：Dataset/Collator → Trainer → DeepSpeed Engine；新增的仅是可观测性与内存管理策略。

## Directory Structure

w:/workspace/VLA & VLN/NewJanusVLN/
├── scripts/
│   ├── zero3.json                  # [MODIFY] 显式化 ZeRO-3 bucket/prefetch/persistence；开启 wall_clock_breakdown；减少峰值/碎片化风险。
│   └── train_h800.sh               # [MODIFY] 增加 CUDA allocator 环境变量与可选的 DS/NCCL 稳健设置；保持现有超参与 ZeRO-3 不变。
└── src/qwen_vl/train/
├── train_vln.py                # [MODIFY] 修正“图片数/视觉token数”日志口径；新增可选同步 empty_cache 回调；（可选）输出更细的 step 分解指标。
└── argument.py                 # [MODIFY] 增加 empty_cache 相关可选参数（steps/阈值/开关），默认不影响现有行为。

## Key Code Structures（仅在需要时）

- 新增 TrainingArguments（建议）：
- `empty_cache_steps: Optional[int]`：每 N step（world-process-zero 触发并 barrier 同步）执行 empty_cache；默认 None/0 表示关闭。
- （可选）`empty_cache_reserved_gb_threshold: Optional[float]`：仅当 reserved/allocated 达到阈值才触发，减少无意义 empty_cache。

## Agent Extensions

- **SubAgent: code-explorer**
- Purpose: 系统性搜索仓库内 DeepSpeed/ZeRO-3 既有实践、Trainer 回调模式与可能的性能开关位置，避免引入不一致新模式。
- Expected outcome: 给出与当前项目风格一致的 ZeRO-3 参数建议与最小改动点清单，确保改动可落地且易回退。