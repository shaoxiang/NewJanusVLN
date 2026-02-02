# NewJanusVLN

Training scaffold modeled after JanusVLN (including VGGT), with support for:
- full instruction text + history/current images as input
- supervised prediction of subinstruction, 2D_description, 3D_description
- action generation (MOVE_FORWARD / TURN_LEFT / TURN_RIGHT / STOP)

The model is trained to output a structured response; all fields contribute to LM loss.

## 🚀 快速开始

### 训练加速（重要）

**VGGT 缓存**可将训练速度提升 **3-5倍**（77.86s/it → 15-20s/it）！

```bash
# 1. 预计算 VGGT 特征（一次性，2-4 小时）
python scripts/precompute_vggt_features.py \
  --model_path /path/to/Qwen2.5-VL-3B-Instruct \
  --vggt_model_path /path/to/VGGT-1B \
  --data_root /path/to/train_data \
  --batch_size 4

# 2. 启用缓存训练
export USE_VGGT_CACHE=true
bash scripts/train_h800.sh
```

详细文档：
- 📘 快速入门：[VGGT_CACHE_QUICKSTART.md](VGGT_CACHE_QUICKSTART.md)
- 📗 详细指南：[docs/VGGT_CACHE_SIMPLIFIED.md](docs/VGGT_CACHE_SIMPLIFIED.md)
- 📙 方案对比：[docs/VGGT_CACHE_COMPARISON.md](docs/VGGT_CACHE_COMPARISON.md)

## Layout

```
NewJanusVLN/
  README.md
  requirements.txt
  scripts/
    train.sh
  src/
    qwen_vl/
      data/
        vln_data.py
      model/
        modeling_qwen2_5_vl.py
        vggt/
      train/
        train_vln.py
```

## Dependencies

Recommended transformers >= 4.50:

```
pip install -r requirements.txt
```

## Data Format

Supported JSON or JSONL. Each sample should include:

Required fields:
- instruction / instruction_text / full_instruction
- subinstruction / sub_instruction / sub_instruction_text
- 2D_description / 2d_description / description_2d
- 3D_description / 3d_description / description_3d
- action / action_label (or int: 0=STOP,1=MOVE_FORWARD,2=TURN_LEFT,3=TURN_RIGHT)
- images or image (list), or history_images + current_image

Image list convention: the last image is the current view, the rest are history.

Example:

```json
{
  "id": "episode_0001/0006.jpg",
  "instruction": "Exit the room and turn left, then walk down the hallway.",
  "subinstruction": "Exit the room and turn left.",
  "2D_description": "Starting the left turn inside a bathroom-like area; view is dominated by a tiled wall with a horizontal towel bar.",
  "3D_description": "Small square mosaic tiles cover the wall; metal towel bar spans horizontally near the top right.",
  "action": "TURN_LEFT",
  "images": [
    "episode_0001/0001.jpg",
    "episode_0001/0002.jpg",
    "episode_0001/0006.jpg"
  ]
}
```

## Train Data Folder Format

If you have episode folders like `train_data/<episode_id>/`, the loader can build samples directly from:
- `milestones_result.json` (instruction_text, subinstructions, completion_checkpoints)
- per-frame files: `step_0000_TURN_RIGHT.png` and `step_0000_TURN_RIGHT.json`
  - `landmark_description` -> 2D_DESCRIPTION
  - `spatial_description` -> 3D_DESCRIPTION

Each frame becomes one training sample with history images selected from earlier frames.

## Training

## Output Format

The model is trained to emit exactly:

```
SUBINSTRUCTION: ...
2D_DESCRIPTION: ...
3D_DESCRIPTION: ...
ACTION: ...
```

For inference, parse the `ACTION:` line to get the action string.

## Notes

- Defaults to Qwen2.5-VL; Qwen2-VL is also supported.
- History length is defined by the number of images in each sample.
- Loss weights are applied per output line using the four `*_weight` flags.

## Janus-style VGGT Training

This is the preferred entrypoint to match JanusVLN's framework and to enable VGGT:

```
export PYTHONPATH=$PWD/src
python -m qwen_vl.train.train_vln \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --vggt_model_path facebook/VGGT-1B/ \
  --train_data_root ./train_data \
  --output_dir /path/to/output \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --model_max_length 163840 \
  --num_train_epochs 1 \
  --subinstruction_weight 1.0 \
  --desc2d_weight 1.0 \
  --desc3d_weight 1.0 \
  --action_weight 1.0
```

Notes:
- `qwen_vl.train.train_vln` uses the JanusVLN model class with VGGT.
- `vggt_model_path` can be a HF repo or local checkpoint.
- For JSON/JSONL annotations, use `--annotation_path` and `--data_path`, or register datasets in `src/qwen_vl/data/__init__.py` and use `--dataset_use`.

### Validation + Logging

To enable validation, either:
- provide a separate validation root with `--val_data_root`, or
- split from training with `--eval_split_ratio` (e.g. 0.05).

To print sample model outputs at eval time:
- set `--eval_num_samples` (default 2) and `--eval_max_new_tokens`.

Example:

```
python -m qwen_vl.train.train_vln \
  --train_data_root ./train_data \
  --val_data_root ./val_data \
  --output_dir ./outputs/vln \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --model_max_length 163840 \
  --num_train_epochs 1 \
  --bf16 True \
  --logging_steps 10 \
  --evaluation_strategy steps \
  --eval_steps 200 \
  --eval_num_samples 2 \
  --eval_max_new_tokens 192
```
