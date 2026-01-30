import logging
import re
import os
import pathlib
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, set_seed

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwen_vl.train.trainer
try:
    from trainer import replace_qwen2_vl_attention_class
except ImportError:
    from qwen_vl.train.trainer import replace_qwen2_vl_attention_class

from qwen_vl.data.vln_data import IGNORE_INDEX, make_supervised_data_module
from qwen_vl.train.argument import ModelArguments, DataArguments, TrainingArguments

from transformers import Qwen2VLForConditionalGeneration

local_rank = None
logger = logging.getLogger(__name__)


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class WeightedLossTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        loss_weights = inputs.pop("loss_weights", None)
        segment_ids = inputs.pop("segment_ids", None)
        outputs = model(**inputs, labels=None)

        if labels is None:
            loss = outputs.loss if hasattr(outputs, "loss") else None
            return (loss, outputs) if return_outputs else loss

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_flat = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        )
        active = shift_labels.view(-1) != IGNORE_INDEX

        weight_flat = None
        if loss_weights is not None:
            shift_weights = loss_weights[..., 1:].contiguous().to(loss_flat.device)
            weight_flat = shift_weights.view(-1)
            denom = weight_flat[active].sum().clamp_min(1.0)
            loss = (loss_flat[active] * weight_flat[active]).sum() / denom
        else:
            loss = loss_flat[active].mean()

        if (
            segment_ids is not None
            and hasattr(self, "state")
            and self.model.training
            and getattr(self.args, "logging_steps", 0) > 0
        ):
            shift_seg = segment_ids[..., 1:].contiguous().view(-1).to(loss_flat.device)
            seg_metrics = {}
            names = ["subinstruction", "2d", "3d", "action"]
            for seg_idx, name in enumerate(names, start=1):
                seg_mask = active & (shift_seg == seg_idx)
                if seg_mask.any():
                    if loss_weights is not None and weight_flat is not None:
                        seg_denom = weight_flat[seg_mask].sum().clamp_min(1.0)
                        seg_loss = (loss_flat[seg_mask] * weight_flat[seg_mask]).sum() / seg_denom
                    else:
                        seg_loss = loss_flat[seg_mask].mean()
                    seg_metrics[f"loss_{name}"] = seg_loss.detach().float().item()
                else:
                    seg_metrics[f"loss_{name}"] = 0.0

            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                seq_lens = attention_mask.sum(dim=1)
                seg_metrics["seq_len_max"] = int(seq_lens.max().item())
                seg_metrics["seq_len_mean"] = float(seq_lens.float().mean().item())

                seq_window = getattr(self, "_seq_len_window", None)
                if seq_window is None:
                    seq_window = deque(maxlen=200)
                    self._seq_len_window = seq_window
                seq_window.extend(seq_lens.detach().cpu().tolist())
                if len(seq_window) >= 10:
                    seq_tensor = torch.tensor(list(seq_window), dtype=torch.float32)
                    seg_metrics["seq_len_p95"] = int(torch.quantile(seq_tensor, 0.95).item())
                    seg_metrics["seq_len_p99"] = int(torch.quantile(seq_tensor, 0.99).item())

            batch_size = inputs.get("input_ids").shape[0]

            # 说明：VLN collator 会把 batch 内所有图片展平成一个列表；
            # `pixel_values.shape[0]` 在你的实现中经常是“视觉 patch/token 数”的口径，不是图片张数。
            image_grid_thw = inputs.get("image_grid_thw")
            if image_grid_thw is not None:
                num_images_total = int(image_grid_thw.shape[0])
                num_images_per_sample = num_images_total / max(batch_size, 1)
                seg_metrics["num_images_per_sample"] = num_images_per_sample

                img_window = getattr(self, "_num_images_window", None)
                if img_window is None:
                    img_window = deque(maxlen=200)
                    self._num_images_window = img_window
                img_window.append(float(num_images_per_sample))
                if len(img_window) >= 10:
                    img_tensor = torch.tensor(list(img_window), dtype=torch.float32)
                    seg_metrics["num_images_per_sample_p95"] = float(torch.quantile(img_tensor, 0.95).item())

                visual_tokens_premerge_total = float(image_grid_thw.prod(dim=1).sum().item())
                visual_tokens_premerge_per_sample = visual_tokens_premerge_total / max(batch_size, 1)
                seg_metrics["visual_tokens_per_sample_premerge"] = visual_tokens_premerge_per_sample

                vc = getattr(getattr(self.model, "config", None), "vision_config", None)
                merge_size = getattr(vc, "spatial_merge_size", 2) if vc is not None else 2
                try:
                    merge_size = int(merge_size)
                except Exception:
                    merge_size = 2
                visual_tokens_postmerge_per_sample = visual_tokens_premerge_per_sample / max(merge_size * merge_size, 1)
                seg_metrics["visual_tokens_per_sample_postmerge_est"] = visual_tokens_postmerge_per_sample

                vis_window = getattr(self, "_visual_tokens_window", None)
                if vis_window is None:
                    vis_window = deque(maxlen=200)
                    self._visual_tokens_window = vis_window
                vis_window.append(float(visual_tokens_premerge_per_sample))
                if len(vis_window) >= 10:
                    vis_tensor = torch.tensor(list(vis_window), dtype=torch.float32)
                    seg_metrics["visual_tokens_per_sample_premerge_p95"] = float(torch.quantile(vis_tensor, 0.95).item())

            pixel_values = inputs.get("pixel_values")
            if pixel_values is not None:
                # 仅用于诊断：表示视觉侧张量第一维的“行数”（通常是 patch/token 行），避免误当图片张数
                seg_metrics["pixel_values_rows_per_sample"] = float(pixel_values.shape[0]) / max(batch_size, 1)

            step = int(self.state.global_step)
            last_step = getattr(self, "_last_segment_log_step", -1)
            if step > 0 and step != last_step and step % self.args.logging_steps == 0:
                self._last_segment_log_step = step
                self.log(seg_metrics)

        return (loss, outputs) if return_outputs else loss


class StepTimeLoggerCallback(transformers.TrainerCallback):
    def __init__(self):
        super().__init__()
        self._step_start = None

    def on_step_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self._step_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or self._step_start is None:
            return
        step_time = time.time() - self._step_start
        logs = {"step_time": step_time}
        if torch.cuda.is_available():
            logs["gpu_max_mem_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            logs["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
        if args.per_device_train_batch_size:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
            else:
                world_size = 1
            logs["samples_per_second_est"] = (
                args.per_device_train_batch_size * world_size
            ) / max(step_time, 1e-6)

        if args.logging_steps > 0 and state.global_step % args.logging_steps == 0:
            trainer = kwargs.get("trainer")
            if trainer is not None:
                trainer.log(logs)


class SyncEmptyCacheCallback(transformers.TrainerCallback):
    """Optional: synchronize `empty_cache()` across ranks to reduce allocator cache flush jitter.

    Default is disabled (empty_cache_steps=0). Keep frequency low (e.g., 50-200) to avoid extra sync overhead.
    """

    def on_step_end(self, args, state, control, **kwargs):
        steps = int(getattr(args, "empty_cache_steps", 0) or 0)
        if steps <= 0:
            return
        if state.global_step <= 0 or (state.global_step % steps) != 0:
            return

        if not torch.cuda.is_available():
            return

        threshold_gb = float(getattr(args, "empty_cache_reserved_gb_threshold", 0.0) or 0.0)
        if threshold_gb > 0.0:
            reserved_gb = torch.cuda.memory_reserved() / (1024**3)
            if reserved_gb < threshold_gb:
                return

        dist_on = torch.distributed.is_available() and torch.distributed.is_initialized()
        if dist_on:
            torch.distributed.barrier()

        try:
            from deepspeed.accelerator import get_accelerator

            get_accelerator().empty_cache()
        except Exception:
            torch.cuda.empty_cache()

        if dist_on:
            torch.distributed.barrier()

        if state.is_world_process_zero:
            trainer = kwargs.get("trainer")
            if trainer is not None:
                trainer.log({"synced_empty_cache": 1, "empty_cache_reserved_gb": torch.cuda.memory_reserved() / (1024**3)})


class ActionWeightSchedulerCallback(transformers.TrainerCallback):
    def __init__(self, data_args, training_args):
        super().__init__()
        self.data_args = data_args
        self.training_args = training_args
        self._weights = None
        self._milestones = None
        self._current = None
        self._parse_schedule()

    def _parse_schedule(self):
        if not self.training_args.action_weight_schedule or not self.training_args.action_weight_milestones:
            return
        try:
            weights = [float(x) for x in self.training_args.action_weight_schedule.split(",")]
            milestones = [float(x) for x in self.training_args.action_weight_milestones.split(",")]
            if len(weights) != 3 or len(milestones) != 2:
                return
            self._weights = weights
            self._milestones = milestones
        except Exception:
            self._weights = None
            self._milestones = None

    def _get_weight(self, progress: float):
        if self._weights is None or self._milestones is None:
            return None
        if progress <= self._milestones[0]:
            return self._weights[0]
        if progress <= self._milestones[1]:
            return self._weights[1]
        return self._weights[2]

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or state.max_steps <= 0:
            return
        target = self._get_weight(state.global_step / state.max_steps)
        if target is None:
            return
        if self._current is None or abs(self._current - target) > 1e-6:
            self._current = target
            self.data_args.action_weight = target
            trainer = kwargs.get("trainer")
            if trainer is not None:
                trainer.log({"action_weight": target})


# Sample prediction logging disabled per request.


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for _, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for _, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for _, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for _, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for _, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for _, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

    for _, p in model.vggt.named_parameters():
        p.requires_grad = False
    for _, p in model.merger.named_parameters():
        p.requires_grad = True


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False

    if attn_implementation.startswith("flash") and not (training_args.bf16 or training_args.fp16):
        logging.warning("Flash attention requires bf16/fp16. Falling back to sdpa.")
        attn_implementation = "sdpa"

    if not torch.distributed.is_initialized() and torch.cuda.device_count() > 1:
        # Avoid DataParallel; keep behavior consistent with single-GPU unless user launches via torchrun.
        training_args._n_gpu = 1
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        logging.warning("Multiple GPUs detected without torchrun; forcing single-GPU to avoid DataParallel issues.")

    torch_dtype = None
    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16

    if "qwen2.5" in model_args.model_name_or_path.lower():
        from qwen_vl.model.modeling_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGenerationForJanusVLN,
        )

        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        setattr(config, "lam", model_args.lam)
        setattr(config, "reference_frame", model_args.reference_frame)
        # vision feature caching (training paradigm improvement when vision towers are frozen)
        setattr(config, "vision_feature_cache", bool(getattr(training_args, "vision_feature_cache", False)))
        setattr(config, "vision_feature_cache_write", bool(getattr(training_args, "vision_feature_cache_write", True)))
        setattr(config, "vision_feature_cache_dir", getattr(training_args, "vision_feature_cache_dir", None))
        setattr(config, "vision_feature_cache_max_entries", int(getattr(training_args, "vision_feature_cache_max_entries", 128)))

        # vggt feature caching (effective when VGGT is frozen; key is per-sample history image_paths)
        setattr(config, "vggt_feature_cache", bool(getattr(training_args, "vggt_feature_cache", False)))
        setattr(config, "vggt_feature_cache_write", bool(getattr(training_args, "vggt_feature_cache_write", True)))
        setattr(config, "vggt_feature_cache_dir", getattr(training_args, "vggt_feature_cache_dir", None))
        setattr(config, "vggt_feature_cache_max_entries", int(getattr(training_args, "vggt_feature_cache_max_entries", 128)))

        # fallback base dir for cache if user doesn't override
        setattr(config, "cache_dir", training_args.cache_dir or training_args.output_dir)
        model = Qwen2_5_VLForConditionalGenerationForJanusVLN.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            vggt_model_path=model_args.vggt_model_path,
        )

        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path
        ).image_processor
        data_args.model_type = "qwen2vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    if data_module.get("train_dataset") is not None:
        rank0_print(f"train_dataset_size={len(data_module['train_dataset'])}")
    if data_module.get("eval_dataset") is not None:
        rank0_print(f"eval_dataset_size={len(data_module['eval_dataset'])}")

    trainer = WeightedLossTrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.add_callback(StepTimeLoggerCallback())
    if int(getattr(training_args, "empty_cache_steps", 0) or 0) > 0:
        trainer.add_callback(SyncEmptyCacheCallback())
    trainer.add_callback(ActionWeightSchedulerCallback(data_args, training_args))

    checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    save_only_model = bool(getattr(training_args, "save_only_model", False))

    def _ckpt_step(p: pathlib.Path) -> int:
        m = re.search(r"checkpoint-(\d+)$", p.name)
        return int(m.group(1)) if m else -1

    last_ckpt = max(checkpoints, key=_ckpt_step) if checkpoints else None
    has_deepspeed_state = False
    if last_ckpt is not None:
        has_deepspeed_state = bool(list(last_ckpt.glob("global_step*"))) or bool(
            list(last_ckpt.glob("**/zero_pp_rank_*_model_states.pt"))
        )

    # If we only save model weights, DeepSpeed checkpoint state is typically absent; do not auto-resume.
    if save_only_model and last_ckpt is not None:
        logging.warning("checkpoint found but save_only_model=True; skipping resume")
        trainer.train()
    elif last_ckpt is not None and (not trainer.deepspeed or has_deepspeed_state):
        logging.info(f"checkpoint found, resume training from {last_ckpt}")
        trainer.train(resume_from_checkpoint=str(last_ckpt))
    else:
        if last_ckpt is not None and trainer.deepspeed and not has_deepspeed_state:
            logging.warning("checkpoint found but no DeepSpeed state; skipping resume")
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    source_path = os.path.join(model_args.model_name_or_path, "chat_template.json")
    template_path = os.path.join(training_args.output_dir, "chat_template.json")
    if os.path.exists(source_path):
        shutil.copy2(source_path, template_path)

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
