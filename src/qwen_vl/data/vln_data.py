import copy
import json
import os
import random
import time
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
STEP_PATTERN = re.compile(r"^step_(\d+)_([A-Za-z_]+)\.(png|jpg|jpeg)$")


def read_jsonl(path: str, max_samples: int = -1):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
            if max_samples != -1 and len(samples) >= max_samples:
                break
    return samples


def _get_first(sample: Dict, keys: List[str], default=None):
    for key in keys:
        if key in sample and sample[key] is not None:
            return sample[key]
    return default


def _normalize_text(value, field_name: str):
    if isinstance(value, list):
        if not value:
            raise ValueError(f"Empty list for {field_name}.")
        value = value[0]
    if isinstance(value, str):
        return value.strip()
    return value


def _normalize_action(action_value):
    if isinstance(action_value, str):
        return action_value.strip()
    if isinstance(action_value, int):
        action_map = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
        if 0 <= action_value < len(action_map):
            return action_map[action_value]
    raise ValueError(f"Unknown action format: {action_value}")


def _resolve_image_paths(sample: Dict, data_path: str) -> List[str]:
    if "images" in sample:
        images = sample["images"]
    elif "image" in sample:
        images = sample["image"]
    elif "history_images" in sample and "current_image" in sample:
        images = list(sample["history_images"]) + [sample["current_image"]]
    else:
        raise ValueError("Missing image fields. Expected images/image or history_images+current_image.")

    if isinstance(images, str):
        images = [images]

    resolved = []
    for img in images:
        if os.path.isabs(img):
            resolved.append(img)
        else:
            resolved.append(os.path.join(data_path, img))
    return resolved


def _parse_step_filename(filename: str):
    match = STEP_PATTERN.match(filename)
    if not match:
        return None
    frame_idx = int(match.group(1))
    action = match.group(2).upper()
    return frame_idx, action


def _build_subinstruction_by_frame(
    subinstructions: List[str],
    checkpoints: List[Dict],
    total_frames: int,
) -> List[str]:
    if total_frames <= 0:
        return []
    if not subinstructions:
        return [""] * total_frames

    checkpoint_indices = []
    for checkpoint in checkpoints:
        try:
            checkpoint_indices.append(int(checkpoint["checkpoint_frame_index"]))
        except Exception:
            continue
    checkpoint_indices.sort()

    sub_by_frame = [""] * total_frames
    for i, sub in enumerate(subinstructions):
        if i == 0:
            start = 0
        elif i - 1 < len(checkpoint_indices):
            start = checkpoint_indices[i - 1] + 1
        else:
            start = 0
        end = checkpoint_indices[i] if i < len(checkpoint_indices) else total_frames - 1
        start = max(start, 0)
        end = min(end, total_frames - 1)
        if end < start:
            continue
        for f in range(start, end + 1):
            sub_by_frame[f] = sub

    last_sub = subinstructions[-1]
    for i in range(total_frames):
        if not sub_by_frame[i]:
            sub_by_frame[i] = last_sub
    return sub_by_frame


def _load_train_data_root(train_data_root: str, max_history_images: int):
    list_data_dict = []
    if not os.path.isdir(train_data_root):
        raise ValueError(f"train_data_root is not a directory: {train_data_root}")

    episode_dirs = [
        os.path.join(train_data_root, d)
        for d in os.listdir(train_data_root)
        if not d.startswith(".") and os.path.isdir(os.path.join(train_data_root, d))
    ]
    episode_dirs.sort()

    for episode_dir in episode_dirs:
        milestone_path = os.path.join(episode_dir, "milestones_result.json")
        if not os.path.exists(milestone_path):
            continue
        with open(milestone_path, "r", encoding="utf-8") as f:
            milestone = json.load(f)

        instruction = milestone.get("instruction_text", "")
        subinstructions = milestone.get("subinstructions", [])
        checkpoints = milestone.get("completion_checkpoints", [])

        step_entries = []
        for fname in os.listdir(episode_dir):
            parsed = _parse_step_filename(fname)
            if parsed is None:
                continue
            frame_idx, action = parsed
            img_path = os.path.join(episode_dir, fname)
            json_path = os.path.splitext(img_path)[0] + ".json"
            step_entries.append((frame_idx, action, img_path, json_path))

        if not step_entries:
            continue

        step_entries.sort(key=lambda x: x[0])
        max_frame_idx = step_entries[-1][0]
        total_frames = max_frame_idx + 1

        sub_by_frame = _build_subinstruction_by_frame(
            subinstructions=subinstructions,
            checkpoints=checkpoints,
            total_frames=total_frames,
        )

        image_paths = [entry[2] for entry in step_entries]
        for j, (frame_idx, action, img_path, json_path) in enumerate(step_entries):
            if not os.path.exists(json_path):
                continue
            with open(json_path, "r", encoding="utf-8") as f:
                desc = json.load(f)
            desc_2d = desc.get("landmark_description", "")
            desc_3d = desc.get("spatial_description", "")
            if not desc_2d or not desc_3d:
                continue

            if j <= max_history_images:
                idxs = list(range(j + 1))
                if len(idxs) < max_history_images + 1:
                    pad_count = max_history_images + 1 - len(idxs)
                    idxs = [idxs[0]] * pad_count + idxs
            else:
                idxs = np.linspace(0, j, max_history_images + 1, dtype=int).tolist()
            sampled_imgs = [image_paths[idx] for idx in idxs]

            subinstruction = ""
            if 0 <= frame_idx < len(sub_by_frame):
                subinstruction = sub_by_frame[frame_idx]
            if not subinstruction and subinstructions:
                subinstruction = subinstructions[-1]

            sample = {
                "id": f"{os.path.basename(episode_dir)}/{os.path.basename(img_path)}",
                "instruction": instruction,
                "subinstruction": subinstruction,
                "2D_description": desc_2d,
                "3D_description": desc_3d,
                "action": action,
                "images": sampled_imgs,
                "tag": "vln",
            }
            list_data_dict.append(sample)

    return list_data_dict


def _build_conversation(
    instruction: str,
    subinstruction: str,
    desc_2d: str,
    desc_3d: str,
    action: str,
    num_history_images: int,
) -> List[Dict[str, str]]:
    history_tags = DEFAULT_IMAGE_TOKEN * num_history_images
    prompt = (
        "You are a visual language navigation model, and your should go to the locations to complete the given task. "
        "Compare the observation and instruction to infer your current progress, and then select the correct direction "
        "from the candidates to go to the target location and finish the task.\n"
        f"This is your historical observation:{history_tags}\n"
        "This is your current observation:<image>\n"
        f"Your task is to {instruction}\n"
        "First, predict the subinstruction for the current view. Then describe the view in 2D and 3D. "
        "Finally, choose one of the following actions:\n"
        "MOVE_FORWARD\n"
        "TURN_LEFT\n"
        "TURN_RIGHT\n"
        "STOP.\n"
        "Respond in the following format:\n"
        "SUBINSTRUCTION: ...\n"
        "2D_DESCRIPTION: ...\n"
        "3D_DESCRIPTION: ...\n"
        "ACTION: ..."
    )
    target = (
        f"SUBINSTRUCTION: {subinstruction}\n"
        f"2D_DESCRIPTION: {desc_2d}\n"
        f"3D_DESCRIPTION: {desc_3d}\n"
        f"ACTION: {action}"
    )
    return [
        {"from": "human", "value": prompt},
        {"from": "gpt", "value": target},
    ]


def _find_subsequence(sequence: List[int], pattern: List[int]) -> int:
    if not pattern or len(pattern) > len(sequence):
        return -1
    for i in range(len(sequence) - len(pattern) + 1):
        if sequence[i : i + len(pattern)] == pattern:
            return i
    return -1


def _build_loss_weights(
    tokenizer: transformers.PreTrainedTokenizer,
    input_ids: List[int],
    labels: torch.Tensor,
    segment_texts: List[str],
    segment_weights: List[float],
) -> torch.Tensor:
    weights = torch.zeros(len(input_ids), dtype=torch.float)
    label_mask = labels != IGNORE_INDEX

    target_text = "\n".join(segment_texts)
    target_token_ids = tokenizer.encode(target_text, add_special_tokens=False)
    start_idx = _find_subsequence(input_ids, target_token_ids)
    if start_idx == -1:
        label_positions = label_mask.nonzero(as_tuple=False)
        if label_positions.numel() == 0:
            return label_mask.float()
        start_idx = label_positions[0].item()

    prefix = ""
    offsets = []
    for i, segment in enumerate(segment_texts):
        if i < len(segment_texts) - 1:
            prefix += segment + "\n"
        else:
            prefix += segment
        offsets.append(len(tokenizer.encode(prefix, add_special_tokens=False)))

    prev = 0
    for seg_idx, end in enumerate(offsets):
        seg_start = start_idx + prev
        seg_end = start_idx + end
        weights[seg_start:seg_end] = segment_weights[seg_idx]
        prev = end

    fallback_mask = label_mask & (weights == 0)
    weights[fallback_mask] = 1.0
    return weights


def _build_segment_ids(
    tokenizer: transformers.PreTrainedTokenizer,
    input_ids: List[int],
    labels: torch.Tensor,
    segment_texts: List[str],
) -> torch.Tensor:
    seg_ids = torch.zeros(len(input_ids), dtype=torch.long)
    label_mask = labels != IGNORE_INDEX
    label_positions = label_mask.nonzero(as_tuple=False).view(-1)
    if label_positions.numel() == 0:
        return seg_ids

    label_tokens = labels[label_mask].tolist()
    markers = ["SUBINSTRUCTION:", "2D_DESCRIPTION:", "3D_DESCRIPTION:", "ACTION:"]
    marker_positions = []
    search_start = 0
    for marker in markers:
        marker_tokens = tokenizer.encode(marker, add_special_tokens=False)
        if not marker_tokens:
            break
        best_pos = None
        best_len = None
        for prefix in ("", "\n", " ", "\n\n"):
            variant = prefix + marker
            variant_tokens = tokenizer.encode(variant, add_special_tokens=False)
            if not variant_tokens:
                continue
            rel_start = _find_subsequence(label_tokens[search_start:], variant_tokens)
            if rel_start == -1:
                continue
            abs_start = search_start + rel_start
            marker_start = abs_start + (len(variant_tokens) - len(marker_tokens))
            if best_pos is None or marker_start < best_pos:
                best_pos = marker_start
                best_len = len(marker_tokens)
        if best_pos is None:
            break
        marker_positions.append((best_pos, best_len))
        search_start = best_pos + best_len

    if not marker_positions:
        return seg_ids

    for idx, (start, marker_len) in enumerate(marker_positions):
        end = marker_positions[idx + 1][0] if idx + 1 < len(marker_positions) else len(label_tokens)
        seg_ids[label_positions[start:end]] = idx + 1

    return seg_ids


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw: List = None,
    visual_type: str = "image",
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    tokenizer.chat_template = chat_template

    if grid_thw is None:
        grid_thw = []

    visual_replicate_index = 0
    input_ids, targets = [], []

    for source in sources:
        if roles.get(source[0]["from"], source[0]["from"]) != roles["human"]:
            source = source[1:]

        input_id, target = [], []
        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            role = conv.get("role") or conv.get("from")
            content = conv.get("content") or conv.get("value")
            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|{visual_type}_pad|>" * grid_thw[visual_replicate_index]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        if len(input_id) != len(target):
            raise ValueError(f"Token/label length mismatch: {len(input_id)} != {len(target)}")
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(torch.utils.data.Dataset):
    """Dataset for supervised fine-tuning with VGGT support."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
        list_data_dict: Optional[List[Dict]] = None,
        shuffle: Optional[bool] = None,
        max_samples_override: Optional[int] = None,
    ):
        super().__init__()

        if list_data_dict is None:
            if data_args.train_data_root:
                list_data_dict = _load_train_data_root(
                    data_args.train_data_root, data_args.max_history_images
                )
            else:
                if data_args.annotation_path:
                    dataset_list = [
                        {
                            "annotation_path": data_args.annotation_path,
                            "data_path": data_args.data_path or "",
                            "sampling_rate": 1.0,
                            "tag": "vln",
                        }
                    ]
                else:
                    dataset = data_args.dataset_use.split(",") if data_args.dataset_use else []
                    if not dataset:
                        raise ValueError(
                            "Specify --train_data_root, --dataset_use, or --annotation_path."
                        )
                    dataset_list = data_list(dataset)

                list_data_dict = []
                for dataset in dataset_list:
                    ann_path = dataset["annotation_path"]
                    file_format = ann_path.split(".")[-1]
                    if file_format == "jsonl":
                        annotations = read_jsonl(ann_path, max_samples=data_args.max_samples)
                    else:
                        with open(ann_path, "r", encoding="utf-8") as f:
                            annotations = json.load(f)
                    sampling_rate = dataset.get("sampling_rate", 1.0)
                    if sampling_rate < 1.0:
                        annotations = random.sample(
                            annotations, max(1, int(len(annotations) * sampling_rate))
                        )
                    for ann in annotations:
                        ann["data_path"] = dataset.get("data_path", "")
                        ann["tag"] = dataset.get("tag", "vln")
                    list_data_dict += annotations

        if shuffle is None:
            shuffle = data_args.shuffle
        if shuffle:
            random.shuffle(list_data_dict)

        max_samples = data_args.max_samples if max_samples_override is None else max_samples_override
        if max_samples != -1 and len(list_data_dict) > max_samples:
            list_data_dict = list_data_dict[:max_samples]

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels

        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

    def __len__(self):
        return len(self.list_data_dict)

    def process_image_unified_vggt(self, image_file: str):
        # this function reshapes the image to the width size of 518
        image_processor = copy.deepcopy(self.data_args.image_processor)
        from qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images
        images = load_and_preprocess_images([image_file], mode="pad")
        images_vggt = copy.deepcopy(images[0])
        merge_size: int = getattr(image_processor, "merge_size")
        patch_size: int = getattr(image_processor, "patch_size")
        _, height, width = images[0].shape
        if (width // patch_size) % merge_size > 0:
            width = width - (width // patch_size) % merge_size * patch_size
        if (height // patch_size) % merge_size > 0:
            height = height - (height // patch_size) % merge_size * patch_size
        images = images[:, :, :height, :width]
        target_pixels = int(height * width)
        visual_processed = image_processor(
            images,
            return_tensors="pt",
            do_rescale=False,
            min_pixels=target_pixels,
            max_pixels=target_pixels,
        )
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        target_h = int(grid_thw[1].item()) * patch_size
        target_w = int(grid_thw[2].item()) * patch_size
        if images_vggt.shape[-2] != target_h or images_vggt.shape[-1] != target_w:
            images_vggt = F.interpolate(
                images_vggt.unsqueeze(0),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        # return image_tensor, grid_thw
        return {
            "pixel_values": image_tensor,
            "image_grid_thw": grid_thw,
            "images_vggt": images_vggt
        }

    def _extract_fields(self, sample: Dict):
        instruction = _get_first(
            sample,
            ["instruction", "instruction_text", "full_instruction"],
            default=None,
        )
        if instruction is None and "instructions" in sample:
            instruction = sample["instructions"][0]
        if instruction is None:
            raise ValueError("Missing instruction field in sample.")
        instruction = _normalize_text(instruction, "instruction")

        subinstruction = _get_first(
            sample,
            ["subinstruction", "sub_instruction", "sub_instruction_text"],
            default=None,
        )
        if subinstruction is None:
            raise ValueError("Missing subinstruction field in sample.")
        subinstruction = _normalize_text(subinstruction, "subinstruction")

        desc_2d = _get_first(
            sample,
            ["2D_description", "2d_description", "description_2d"],
            default=None,
        )
        if desc_2d is None:
            raise ValueError("Missing 2D_description field in sample.")
        desc_2d = _normalize_text(desc_2d, "2D_description")

        desc_3d = _get_first(
            sample,
            ["3D_description", "3d_description", "description_3d"],
            default=None,
        )
        if desc_3d is None:
            raise ValueError("Missing 3D_description field in sample.")
        desc_3d = _normalize_text(desc_3d, "3D_description")

        action = _get_first(sample, ["action", "action_label"], default=None)
        if action is None and "actions" in sample:
            action = sample["actions"]
        if action is None:
            raise ValueError("Missing action field in sample.")
        action = _normalize_action(action)

        images = _resolve_image_paths(sample, sample.get("data_path", ""))
        if len(images) < 1:
            raise ValueError("No images found in sample.")
        return instruction, subinstruction, desc_2d, desc_3d, action, images

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        for attempt in range(num_base_retries):
            try:
                return self._get_item(i)
            except Exception as e:
                print(f"[Try #{attempt}] Failed to fetch sample {i}: {e}")
                time.sleep(1)
        return self._get_item(i)

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sample = self.list_data_dict[i]
        instruction, subinstruction, desc_2d, desc_3d, action, images = self._extract_fields(sample)

        num_history = max(len(images) - 1, 0)
        conversations = _build_conversation(
            instruction=instruction,
            subinstruction=subinstruction,
            desc_2d=desc_2d,
            desc_3d=desc_3d,
            action=action,
            num_history_images=num_history,
        )

        image_tensors = []
        grid_thw = []
        images_vggt = []
        for image_path in images:
            ret = self.process_image_unified_vggt(image_path)
            image_tensors.append(ret["pixel_values"])
            images_vggt.append(ret["images_vggt"])
            grid_thw.append(ret["image_grid_thw"])

        merge_size = getattr(self.data_args.image_processor, "merge_size", 2)
        grid_thw_merged = [
            thw.prod().item() // (merge_size**2) for thw in grid_thw
        ]

        sources = copy.deepcopy([conversations])
        data_dict = preprocess_qwen_2_visual(
            sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="image"
        )
        position_ids, _ = self.get_rope_index(
            merge_size,
            data_dict["input_ids"],
            torch.stack(grid_thw, dim=0),
        )

        segment_texts = [
            f"SUBINSTRUCTION: {subinstruction}",
            f"2D_DESCRIPTION: {desc_2d}",
            f"3D_DESCRIPTION: {desc_3d}",
            f"ACTION: {action}",
        ]
        segment_weights = [
            self.data_args.subinstruction_weight,
            self.data_args.desc2d_weight,
            self.data_args.desc3d_weight,
            self.data_args.action_weight,
        ]
        loss_weights = _build_loss_weights(
            tokenizer=self.tokenizer,
            input_ids=data_dict["input_ids"][0].tolist(),
            labels=data_dict["labels"][0],
            segment_texts=segment_texts,
            segment_weights=segment_weights,
        )
        segment_ids = _build_segment_ids(
            tokenizer=self.tokenizer,
            input_ids=data_dict["input_ids"][0].tolist(),
            labels=data_dict["labels"][0],
            segment_texts=segment_texts,
        )

        data_dict = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
            position_ids=position_ids,
            pixel_values=image_tensors,
            image_grid_thw=grid_thw,
            images_vggt=images_vggt,
            tag=sample.get("tag", "vln"),
            loss_weights=loss_weights,
            segment_ids=segment_ids,
        )
        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)
    return torch.cat(padded_tensors, dim=1)


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        loss_weights = []
        segment_ids = []
        for instance in instances:
            lw = instance.get("loss_weights")
            if lw is None:
                labels_i = instance["labels"]
                lw = (labels_i != IGNORE_INDEX).float()
            loss_weights.append(lw)
            seg = instance.get("segment_ids")
            if seg is None:
                seg = torch.zeros_like(instance["labels"], dtype=torch.long)
            segment_ids.append(seg)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        loss_weights = torch.nn.utils.rnn.pad_sequence(
            loss_weights, batch_first=True, padding_value=0.0
        )
        segment_ids = torch.nn.utils.rnn.pad_sequence(
            segment_ids, batch_first=True, padding_value=0
        )

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        loss_weights = loss_weights[:, : self.tokenizer.model_max_length]
        segment_ids = segment_ids[:, : self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            position_ids=position_ids,
            loss_weights=loss_weights,
            segment_ids=segment_ids,
        )

        images = []
        grid_thw = []
        for instance in instances:
            if "pixel_values" in instance:
                images.extend(instance["pixel_values"])
                grid_thw.extend(instance["image_grid_thw"])

        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            batch["image_grid_thw"] = torch.stack(grid_thw, dim=0)
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None

        if "images_vggt" in instances[0]:
            images_vggt = [torch.stack(instance["images_vggt"]) for instance in instances]
            batch["images_vggt"] = images_vggt
        batch["tag"] = instances[0].get("tag", "vln")
        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    eval_dataset = None
    if data_args.val_data_root or data_args.eval_split_ratio > 0:
        if data_args.val_data_root:
            eval_args = copy.deepcopy(data_args)
            eval_args.train_data_root = data_args.val_data_root
            eval_dataset = LazySupervisedDataset(
                tokenizer=tokenizer,
                data_args=eval_args,
                shuffle=False,
                max_samples_override=data_args.eval_max_samples,
            )
        else:
            total = len(train_dataset.list_data_dict)
            split = max(1, int(total * data_args.eval_split_ratio))
            indices = list(range(total))
            rng = random.Random(data_args.eval_split_seed)
            rng.shuffle(indices)
            eval_list = [train_dataset.list_data_dict[i] for i in indices[:split]]
            train_list = [train_dataset.list_data_dict[i] for i in indices[split:]]
            train_dataset.list_data_dict = train_list
            eval_dataset = LazySupervisedDataset(
                tokenizer=tokenizer,
                data_args=data_args,
                list_data_dict=eval_list,
                shuffle=False,
                max_samples_override=data_args.eval_max_samples,
            )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
