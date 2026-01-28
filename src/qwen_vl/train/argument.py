import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    vggt_model_path: str = field(default="facebook/VGGT-1B/")
    lam: float = field(default=0.2)
    distill_loss_weight: float = field(default=1.0)
    reference_frame: str = field(default="last")


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    annotation_path: Optional[str] = field(
        default=None,
        metadata={"help": "Direct path to JSON/JSONL annotation file"},
    )
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Root folder for image files when using --annotation_path"},
    )
    train_data_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root folder containing episode subfolders with milestones_result.json"},
    )
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    max_samples: int = field(default=-1)
    shuffle: bool = field(default=True)
    max_history_images: int = field(
        default=2, metadata={"help": "Max number of history images per sample"}
    )
    val_data_root: Optional[str] = field(
        default=None,
        metadata={"help": "Optional validation root with episode subfolders"},
    )
    eval_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Split ratio for validation when val_data_root is not set"},
    )
    eval_split_seed: int = field(default=42)
    eval_max_samples: int = field(
        default=-1, metadata={"help": "Max eval samples when building eval dataset"}
    )
    eval_num_samples: int = field(
        default=2, metadata={"help": "Num eval samples to log with generation"}
    )
    eval_max_new_tokens: int = field(
        default=192, metadata={"help": "Max new tokens to generate for eval previews"}
    )
    subinstruction_weight: float = field(
        default=1.0, metadata={"help": "Loss weight for SUBINSTRUCTION tokens"}
    )
    desc2d_weight: float = field(
        default=1.0, metadata={"help": "Loss weight for 2D_DESCRIPTION tokens"}
    )
    desc3d_weight: float = field(
        default=1.0, metadata={"help": "Loss weight for 3D_DESCRIPTION tokens"}
    )
    action_weight: float = field(
        default=1.0, metadata={"help": "Loss weight for ACTION tokens"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
