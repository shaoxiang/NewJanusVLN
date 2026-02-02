#!/usr/bin/env python
"""
Precompute VGGT visual features for training acceleration.

This script processes all training images through the frozen VGGT encoder
and saves the features next to the image files. During training, features 
are loaded directly, bypassing the expensive VGGT forward pass (3-5x speedup).

Cache files are stored as: <image_path>.vggt_cache.pt

Usage:
    python scripts/precompute_vggt_features.py \
        --model_path /path/to/model \
        --vggt_model_path /path/to/vggt \
        --data_root /path/to/train_data \
        --batch_size 4 \
        --num_workers 4
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationForJanusVLN


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute VGGT features")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen2.5-VL model")
    parser.add_argument("--vggt_model_path", type=str, required=True, help="Path to VGGT model")
    parser.add_argument("--data_root", type=str, required=True, help="Path to training data root")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--max_samples", type=int, default=-1, help="Max samples to process (-1 for all)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--skip_existing", action="store_true", help="Skip already cached features")
    parser.add_argument("--verify", action="store_true", help="Verify cached features after creation")
    return parser.parse_args()


def get_cache_path(image_path: str) -> str:
    """Get cache file path for an image (stored next to the image)."""
    return f"{image_path}.vggt_cache.pt"


def collect_all_images(data_root: str) -> List[Dict]:
    """Collect all images from training data."""
    print(f"[INFO] Scanning data root: {data_root}")
    
    # Find all JSONL files
    jsonl_files = list(Path(data_root).rglob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No .jsonl files found in {data_root}")
    
    print(f"[INFO] Found {len(jsonl_files)} JSONL files")
    
    all_samples = []
    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    all_samples.append(sample)
                except Exception as e:
                    print(f"[WARN] Failed to parse line in {jsonl_file}: {e}")
    
    print(f"[INFO] Total samples: {len(all_samples)}")
    
    # Extract all unique image paths
    image_map = {}  # path -> metadata
    for sample_idx, sample in enumerate(all_samples):
        # Extract images
        images = []
        if "images" in sample:
            images = sample["images"] if isinstance(sample["images"], list) else [sample["images"]]
        elif "image" in sample:
            images = [sample["image"]] if isinstance(sample["image"], str) else sample["image"]
        elif "history_images" in sample and "current_image" in sample:
            images = list(sample["history_images"]) + [sample["current_image"]]
        
        # Resolve paths
        data_path = sample.get("data_path", "")
        for img_rel in images:
            if os.path.isabs(img_rel):
                img_full = img_rel
            else:
                img_full = os.path.join(data_root, data_path, img_rel)
            
            if not os.path.exists(img_full):
                print(f"[WARN] Image not found: {img_full}")
                continue
            
            if img_full not in image_map:
                image_map[img_full] = {
                    "path": img_full,
                    "sample_indices": [sample_idx],
                }
            else:
                image_map[img_full]["sample_indices"].append(sample_idx)
    
    image_list = list(image_map.values())
    print(f"[INFO] Unique images to process: {len(image_list)}")
    
    return image_list


def process_images_batch(
    model,
    processor,
    image_batch: List[Dict],
    device: str,
    skip_existing: bool,
) -> int:
    """Process a batch of images and save features."""
    # Filter out already cached
    to_process = []
    for img_meta in image_batch:
        cache_path = get_cache_path(img_meta['path'])
        if skip_existing and os.path.exists(cache_path):
            continue
        to_process.append(img_meta)
    
    if not to_process:
        return 0
    
    # Load images
    pil_images = []
    valid_metas = []
    for img_meta in to_process:
        try:
            pil_img = Image.open(img_meta["path"]).convert("RGB")
            pil_images.append(pil_img)
            valid_metas.append(img_meta)
        except Exception as e:
            print(f"[ERROR] Failed to load {img_meta['path']}: {e}")
    
    if not pil_images:
        return 0
    
    # Process with VGGT (mimicking modeling_qwen2_5_vl.py:2096-2127)
    try:
        # Use processor to get VGGT-compatible tensors
        # Note: This requires processor to have `process_vggt_images` or equivalent
        # For now, we'll use a simplified approach matching your model's expectations
        
        images_vggt = []
        for pil_img in pil_images:
            # Resize to VGGT input size (you may need to adjust based on config)
            # Assuming 518x518 as in your original code
            img_resized = pil_img.resize((518, 518), Image.BILINEAR)
            img_tensor = torch.tensor(processor.image_processor.preprocess(
                img_resized, return_tensors="pt"
            )["pixel_values"])
            images_vggt.append(img_tensor.squeeze(0))
        
        images_vggt = torch.stack(images_vggt).to(device)
        
        # Run VGGT aggregator (must match model's forward logic)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Initialize past_key_values for batch processing
                past_key_values = [None] * model.vggt.aggregator.depth
                
                features_batch = []
                for img_tensor in images_vggt:
                    # Process single image through temporal aggregator
                    # Note: For history images, we process frame-by-frame
                    # Here we assume single-frame per image
                    img_input = img_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
                    
                    aggregator_output = model.vggt.aggregator(
                        img_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                        past_frame_idx=0
                    )
                    
                    if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                        aggregated_tokens, patch_start_idx, _ = aggregator_output
                    else:
                        aggregated_tokens, patch_start_idx = aggregator_output
                    
                    features = aggregated_tokens[-2][0, :, patch_start_idx:]
                    features_batch.append(features.cpu())
        
        # Save features next to images
        for features, img_meta in zip(features_batch, valid_metas):
            cache_path = get_cache_path(img_meta["path"])
            torch.save({
                "features": features,
                "path": img_meta["path"],
            }, cache_path)
        
        return len(features_batch)
    
    except Exception as e:
        print(f"[ERROR] Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 0


def verify_cache(image_list: List[Dict]):
    """Verify cached features are valid."""
    print("\n[INFO] Verifying cached features...")
    valid = 0
    invalid = []
    
    for img_meta in tqdm(image_list, desc="Verifying"):
        cache_path = get_cache_path(img_meta["path"])
        if not os.path.exists(cache_path):
            invalid.append(img_meta["path"])
            continue
        
        try:
            data = torch.load(cache_path, map_location="cpu")
            assert "features" in data
            assert data["features"].dim() >= 2
            valid += 1
        except Exception as e:
            print(f"[ERROR] Invalid cache: {cache_path}: {e}")
            invalid.append(img_meta["path"])
    
    print(f"[INFO] Verification: {valid}/{len(image_list)} valid, {len(invalid)} invalid")
    if invalid:
        print(f"[WARN] Invalid cached files (first 10): {invalid[:10]}")


def main():
    args = parse_args()
    
    print("=" * 80)
    print("VGGT Feature Precomputation")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"VGGT model path: {args.vggt_model_path}")
    print(f"Data root: {args.data_root}")
    print(f"Cache mode: Store next to images (*.vggt_cache.pt)")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Collect all images
    image_list = collect_all_images(args.data_root)
    if args.max_samples > 0:
        image_list = image_list[:args.max_samples]
        print(f"[INFO] Limited to {args.max_samples} images")
    
    # Load model
    print(f"\n[INFO] Loading model from {args.model_path}...")
    model = Qwen2_5_VLForConditionalGenerationForJanusVLN.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        vggt_model_path=args.vggt_model_path,
    )
    model.eval()
    model.vggt.eval()
    
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    print(f"[INFO] Model loaded successfully")
    
    # Process images
    print(f"\n[INFO] Processing {len(image_list)} images...")
    processed = 0
    
    for i in tqdm(range(0, len(image_list), args.batch_size), desc="Processing"):
        batch = image_list[i:i + args.batch_size]
        n = process_images_batch(
            model, processor, batch, args.device, args.skip_existing
        )
        processed += n
    
    print(f"\n[SUCCESS] Processed {processed} images")
    print(f"[INFO] Cache files stored next to images as: <image_path>.vggt_cache.pt")
    
    # Verify if requested
    if args.verify:
        verify_cache(image_list)
    
    # Write manifest
    manifest_path = os.path.join(args.data_root, "vggt_cache_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "total_images": len(image_list),
            "processed_images": processed,
            "model_path": args.model_path,
            "vggt_model_path": args.vggt_model_path,
            "data_root": args.data_root,
            "cache_format": "<image_path>.vggt_cache.pt",
        }, f, indent=2)
    print(f"[INFO] Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
