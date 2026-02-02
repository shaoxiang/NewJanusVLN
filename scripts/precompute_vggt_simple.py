#!/usr/bin/env python
"""
Simplified VGGT Feature Precomputation Script

This script precomputes VGGT visual features and saves them next to images.
Cache files are stored as: <image_path>.vggt_cache.pt

Usage:
    python scripts/precompute_vggt_simple.py \
        --vggt_model_path /path/to/VGGT-1B \
        --data_root /path/to/train_data \
        --batch_size 8
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qwen_vl.model.vggt.models.vggt import VGGT


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute VGGT features")
    parser.add_argument("--vggt_model_path", type=str, required=True, help="Path to VGGT model")
    parser.add_argument("--data_root", type=str, required=True, help="Path to training data root")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--skip_existing", action="store_true", help="Skip already cached features")
    parser.add_argument("--max_samples", type=int, default=-1, help="Max images to process (-1 for all)")
    parser.add_argument("--image_list_file", type=str, default=None, help="Pre-generated image list file (for multi-GPU)")
    return parser.parse_args()


def get_cache_path(image_path: str) -> str:
    """Get cache file path for an image."""
    return f"{image_path}.vggt_cache.pt"


def collect_all_images(data_root: str, image_list_file: str = None) -> List[str]:
    """Collect all image paths from training data or pre-generated list."""
    
    # If image list file is provided (multi-GPU mode), read from it
    if image_list_file and os.path.exists(image_list_file):
        print(f"[INFO] Reading image list from: {image_list_file}")
        with open(image_list_file, "r") as f:
            image_list = [line.strip() for line in f if line.strip()]
        print(f"[INFO] Loaded {len(image_list)} images from file")
        return image_list
    
    print(f"[INFO] Scanning data root: {data_root}")
    
    # Find all JSONL files
    jsonl_files = list(Path(data_root).rglob("*.jsonl"))
    if not jsonl_files:
        print(f"[WARN] No .jsonl files found, scanning for images directly...")
        # Fallback: find all images
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            image_paths.extend(Path(data_root).rglob(ext))
        return [str(p) for p in image_paths]
    
    print(f"[INFO] Found {len(jsonl_files)} JSONL files")
    
    # Extract image paths from JSONL
    all_image_paths = set()
    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    
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
                            # Try relative to data_root
                            img_full = os.path.join(data_root, data_path, img_rel)
                            if not os.path.exists(img_full):
                                # Try relative to jsonl file directory
                                img_full = os.path.join(os.path.dirname(jsonl_file), img_rel)
                        
                        if os.path.exists(img_full):
                            all_image_paths.add(img_full)
                        else:
                            print(f"[WARN] Image not found: {img_full}")
                
                except Exception as e:
                    print(f"[WARN] Failed to parse line in {jsonl_file}: {e}")
    
    image_list = sorted(all_image_paths)
    print(f"[INFO] Unique images found: {len(image_list)}")
    
    return image_list


def load_vggt_model(vggt_model_path: str, device: str):
    """Load VGGT model."""
    print(f"[INFO] Loading VGGT model from {vggt_model_path}...")
    
    vggt = VGGT()
    vggt.camera_head = None
    vggt.track_head = None
    
    # Load checkpoint if exists
    ckpt_path = os.path.join(vggt_model_path, "model.pth")
    if os.path.exists(ckpt_path):
        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        vggt.load_state_dict(state_dict, strict=False)
    else:
        print(f"[WARN] No checkpoint found at {ckpt_path}, using random weights")
    
    vggt = vggt.to(device)
    vggt.eval()
    
    for param in vggt.parameters():
        param.requires_grad = False
    
    print(f"[INFO] VGGT model loaded successfully")
    return vggt


def prepare_vggt_image(image_path: str, target_size=(518, 518)):
    """Prepare image for VGGT processing."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    
    # VGGT preprocessing (ImageNet normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img)
    return img_tensor


def process_images_batch(
    vggt_model,
    image_paths: List[str],
    device: str,
    skip_existing: bool,
) -> int:
    """Process a batch of images and save features."""
    # Filter out already cached
    to_process = []
    for img_path in image_paths:
        cache_path = get_cache_path(img_path)
        if skip_existing and os.path.exists(cache_path):
            continue
        to_process.append(img_path)
    
    if not to_process:
        return 0
    
    # Load and prepare images
    images_vggt = []
    valid_paths = []
    
    for img_path in to_process:
        try:
            img_tensor = prepare_vggt_image(img_path)
            images_vggt.append(img_tensor)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"[ERROR] Failed to load {img_path}: {e}")
    
    if not images_vggt:
        return 0
    
    # Stack to batch
    images_batch = torch.stack(images_vggt).to(device)  # [B, C, H, W]
    
    # Process with VGGT
    try:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                features_batch = []
                
                for img_tensor in images_batch:
                    # Process single image through aggregator
                    past_key_values = [None] * vggt_model.aggregator.depth
                    
                    # Add batch and temporal dimensions: [1, 1, C, H, W]
                    img_input = img_tensor.unsqueeze(0).unsqueeze(0)
                    
                    aggregator_output = vggt_model.aggregator(
                        img_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                        past_frame_idx=0
                    )
                    
                    if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                        aggregated_tokens, patch_start_idx, _ = aggregator_output
                    else:
                        aggregated_tokens, patch_start_idx = aggregator_output
                    
                    # Extract features from second-to-last layer
                    features = aggregated_tokens[-2][0, :, patch_start_idx:]
                    features_batch.append(features.cpu())
        
        # Save features
        saved_count = 0
        for features, img_path in zip(features_batch, valid_paths):
            cache_path = get_cache_path(img_path)
            
            try:
                torch.save({
                    "features": features,
                    "path": img_path,
                }, cache_path)
                saved_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to save cache for {img_path}: {e}")
        
        return saved_count
    
    except Exception as e:
        print(f"[ERROR] Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    args = parse_args()
    
    print("=" * 80)
    print("VGGT Feature Precomputation (Simplified)")
    print("=" * 80)
    print(f"VGGT model path: {args.vggt_model_path}")
    print(f"Data root: {args.data_root}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Skip existing: {args.skip_existing}")
    print("=" * 80)
    
    # Collect all images
    image_list = collect_all_images(args.data_root, args.image_list_file)
    
    if not image_list:
        print("[ERROR] No images found!")
        return 1
    
    if args.max_samples > 0:
        image_list = image_list[:args.max_samples]
        print(f"[INFO] Limited to {args.max_samples} images")
    
    # Load VGGT model
    vggt_model = load_vggt_model(args.vggt_model_path, args.device)
    
    # Process images
    print(f"\n[INFO] Processing {len(image_list)} images...")
    total_processed = 0
    
    for i in tqdm(range(0, len(image_list), args.batch_size), desc="Processing"):
        batch = image_list[i:i + args.batch_size]
        n = process_images_batch(vggt_model, batch, args.device, args.skip_existing)
        total_processed += n
    
    print(f"\n[SUCCESS] Processed {total_processed} images")
    print(f"[INFO] Cache files stored as: <image_path>.vggt_cache.pt")
    
    # Write manifest
    manifest_path = os.path.join(args.data_root, "vggt_cache_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "total_images": len(image_list),
            "processed_images": total_processed,
            "vggt_model_path": args.vggt_model_path,
            "data_root": args.data_root,
            "cache_format": "<image_path>.vggt_cache.pt",
        }, f, indent=2)
    print(f"[INFO] Manifest saved to {manifest_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
