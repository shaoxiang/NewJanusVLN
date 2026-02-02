import logging
import re
import os
import pathlib
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, set_seed

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwen_vl.train.trainer
try:
    from .trainer import replace_qwen2_vl_attention_class
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
        action_label = inputs.pop("action_label", None)
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
            loss_text = (loss_flat[active] * weight_flat[active]).sum() / denom
        else:
            loss_text = loss_flat[active].mean()

        # Parallel Action Head Loss
        loss_action = torch.tensor(0.0, device=loss_text.device)
        distill_weight = float(
            getattr(model.config, "distill_loss_weight", getattr(self.args, "distill_loss_weight", 1.0))
        )
        action_logits = getattr(outputs, "action_logits", None)
        if action_label is not None and action_logits is not None:
            loss_action = F.cross_entropy(action_logits, action_label)
            loss = loss_text + distill_weight * loss_action
        else:
            loss = loss_text

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
                    if loss_weights is not None:
                        assert weight_flat is not None
                        seg_denom = weight_flat[seg_mask].sum().clamp_min(1.0)
                        seg_loss = (loss_flat[seg_mask] * weight_flat[seg_mask]).sum() / seg_denom
                    else:
                        seg_loss = loss_flat[seg_mask].mean()
                    seg_metrics[f"loss_{name}"] = seg_loss.detach().float().item()
                else:
                    seg_metrics[f"loss_{name}"] = 0.0

            step = int(self.state.global_step)
            last_step = getattr(self, "_last_segment_log_step", -1)
            if step > 0 and step != last_step and step % self.args.logging_steps == 0:
                self._last_segment_log_step = step
                seg_metrics["loss_action_head"] = loss_action.detach().float().item()
                seg_metrics["loss_text_total"] = loss_text.detach().float().item()
                self.log(seg_metrics)

        return (loss, outputs) if return_outputs else loss


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
    
    # Always train action head
    if hasattr(model, "action_head"):
        for _, p in model.action_head.named_parameters():
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
        setattr(config, "distill_loss_weight", model_args.distill_loss_weight)
        setattr(config, "reference_frame", model_args.reference_frame)
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
    trainer = WeightedLossTrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    source_path = os.path.join(model_args.model_name_or_path, "chat_template.json")
    template_path = os.path.join(training_args.output_dir, "chat_template.json")
    if os.path.exists(source_path):
        shutil.copy2(source_path, template_path)

    model.config.use_cache = True
