#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
Multi-image class embedding optimization for SAM3.

Instead of finding a per-image embedding, this script learns a SINGLE universal
embedding that activates masks across multiple images of the same class.

The key idea: accumulate gradients from multiple image-mask pairs per optimizer
step, forcing the embedding to generalize across all images.

Usage:
    python scripts/optimize_class_embedding.py \
        --image-dir path/to/class_images/ \
        --mask-dir path/to/class_masks/ \
        --seed-text "camouflaged animal" \
        --num-steps 300 \
        --lr 0.01 \
        --output-dir outputs/class_embedding
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize a single token embedding across multiple images of one class"
    )
    parser.add_argument(
        "--image-dir", type=str, required=True,
        help="Directory containing class images",
    )
    parser.add_argument(
        "--mask-dir", type=str, required=True,
        help="Directory containing corresponding binary masks (same filenames)",
    )
    parser.add_argument(
        "--seed-text", type=str, default=None,
        help="Optional seed text to initialize embeddings from",
    )
    parser.add_argument("--num-steps", type=int, default=300, help="Optimization steps")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/class_embedding",
        help="Output directory",
    )
    parser.add_argument("--context-length", type=int, default=16, help="Token positions")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Images per gradient step (all if 0)")
    parser.add_argument("--lambda-mask", type=float, default=5.0)
    parser.add_argument("--lambda-score", type=float, default=2.0)
    parser.add_argument("--lambda-box", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


# Import shared utilities from the single-image script
sys.path.insert(0, os.path.dirname(__file__))
from optimize_text_embedding import (
    load_and_preprocess_image,
    load_target_mask,
    mask_to_bbox_cxcywh,
    init_embedding_from_text,
    init_embedding_random,
    forward_with_embedding,
    select_best_query,
    mask_loss,
    box_giou_loss,
    compute_iou,
    decode_nearest_tokens,
)


def discover_pairs(image_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    """Find matched image-mask pairs by filename."""
    img_exts = {"*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"}
    img_paths = []
    for ext in img_exts:
        img_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    img_paths.sort()

    pairs = []
    for img_path in img_paths:
        stem = Path(img_path).stem
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".bmp"]:
            mask_path = os.path.join(mask_dir, stem + ext)
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
                break
    return pairs


def precompute_features(
    model, pairs: List[Tuple[str, str]], resolution: int, device: torch.device
) -> List[Dict]:
    """Pre-compute image backbone features and target masks for all pairs.

    Since the model is frozen, image features only need to be computed once.
    """
    samples = []
    for img_path, mask_path in pairs:
        img_tensor = load_and_preprocess_image(img_path, resolution, device)
        with torch.no_grad():
            backbone_out = model.backbone.forward_image(img_tensor)

        target_mask_full, target_mask_resized, orig_h, orig_w = load_target_mask(
            mask_path, resolution, device
        )
        target_box = mask_to_bbox_cxcywh(target_mask_resized)

        samples.append({
            "backbone_out": backbone_out,
            "target_mask_resized": target_mask_resized,
            "target_box": target_box,
            "img_path": img_path,
            "mask_path": mask_path,
        })
    return samples


def optimize_class(args):
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Discover image-mask pairs ----
    pairs = discover_pairs(args.image_dir, args.mask_dir)
    print(f"Found {len(pairs)} image-mask pairs")
    if len(pairs) == 0:
        print("Error: No image-mask pairs found!")
        return
    for i, (img, mask) in enumerate(pairs):
        print(f"  [{i}] {Path(img).name} <-> {Path(mask).name}")

    # ---- Load model ----
    print("\nLoading SAM3 model...")
    from sam3 import build_sam3_image_model

    model = build_sam3_image_model(
        device=str(device),
        eval_mode=True,
        checkpoint_path=args.checkpoint_path,
        enable_segmentation=True,
    )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    text_encoder = model.backbone.language_backbone

    # ---- Pre-compute all image features ----
    print("Pre-computing image features (one-time cost)...")
    t0 = time.time()
    samples = precompute_features(model, pairs, args.resolution, device)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ---- Initialize shared embedding ----
    width = text_encoder.encoder.width
    ctx_len = min(args.context_length, text_encoder.context_length)

    if args.seed_text:
        print(f"Initializing from seed text: '{args.seed_text}'")
        opt_embedding = init_embedding_from_text(
            args.seed_text, text_encoder, ctx_len, device
        )
    else:
        print("Initializing with random embeddings")
        opt_embedding = init_embedding_random(ctx_len, width, device)

    opt_embedding = nn.Parameter(opt_embedding)

    # ---- Setup optimizer ----
    optimizer = torch.optim.Adam([opt_embedding], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps, eta_min=args.lr * 0.01
    )

    # ---- Optimization loop ----
    N = len(samples)
    batch_size = args.batch_size if args.batch_size > 0 else N
    print(f"\nOptimizing across {N} images, batch_size={batch_size}, {args.num_steps} steps...")
    print("=" * 70)

    history = {
        "step": [], "loss": [], "mean_iou": [], "per_image_iou": [],
        "mask_loss": [], "score_loss": [], "box_loss": [],
    }
    best_mean_iou = 0.0
    best_embedding = None

    for step in range(args.num_steps):
        optimizer.zero_grad()

        # Sample a mini-batch (or use all)
        if batch_size >= N:
            batch_indices = list(range(N))
        else:
            batch_indices = np.random.choice(N, size=batch_size, replace=False).tolist()

        # Accumulate loss across the batch
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        step_ious = [0.0] * N
        step_mask_loss = 0.0
        step_score_loss = 0.0
        step_box_loss = 0.0
        valid_count = 0

        for idx in batch_indices:
            sample = samples[idx]

            out = forward_with_embedding(
                model=model,
                backbone_out=sample["backbone_out"],
                opt_embedding=opt_embedding,
                text_encoder=text_encoder,
                device=device,
            )

            with torch.no_grad():
                best_q = select_best_query(
                    out,
                    sample["target_mask_resized"],
                    sample["target_box"],
                    args.resolution,
                )

            pred_mask = out["pred_masks"][0, best_q]
            pred_logits = out["pred_logits"]
            pred_boxes = out["pred_boxes"]
            if pred_logits.dim() == 3:
                pred_logit = pred_logits[0, best_q, 0]
                pred_box = pred_boxes[0, best_q]
            else:
                pred_logit = pred_logits[-1, 0, best_q, 0]
                pred_box = pred_boxes[-1, 0, best_q]

            pred_h, pred_w = pred_mask.shape
            target_for_loss = F.interpolate(
                sample["target_mask_resized"].unsqueeze(0).unsqueeze(0),
                size=(pred_h, pred_w),
                mode="nearest",
            ).squeeze(0).squeeze(0)

            l_mask = mask_loss(pred_mask, target_for_loss)
            l_score = F.binary_cross_entropy_with_logits(
                pred_logit, torch.ones_like(pred_logit)
            )
            l_box = (
                F.l1_loss(pred_box, sample["target_box"])
                + box_giou_loss(pred_box, sample["target_box"])
            )

            img_loss = (
                args.lambda_mask * l_mask
                + args.lambda_score * l_score
                + args.lambda_box * l_box
            )

            if not (torch.isnan(img_loss) or torch.isinf(img_loss)):
                total_loss = total_loss + img_loss / len(batch_indices)
                valid_count += 1
                step_mask_loss += l_mask.item()
                step_score_loss += l_score.item()
                step_box_loss += l_box.item()

            with torch.no_grad():
                step_ious[idx] = compute_iou(pred_mask, target_for_loss)

        # Backward + step
        if valid_count > 0 and not (torch.isnan(total_loss) or torch.isinf(total_loss)):
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([opt_embedding], max_norm=1.0)
            optimizer.step()
        else:
            optimizer.zero_grad()
        scheduler.step()

        # NaN cleanup
        with torch.no_grad():
            nan_mask = torch.isnan(opt_embedding)
            if nan_mask.any():
                opt_embedding.data[nan_mask] = 0.0

        # Full eval on all images at log intervals
        if step % args.log_interval == 0 or step == args.num_steps - 1:
            all_ious = []
            with torch.no_grad():
                for idx, sample in enumerate(samples):
                    out_eval = forward_with_embedding(
                        model=model,
                        backbone_out=sample["backbone_out"],
                        opt_embedding=opt_embedding,
                        text_encoder=text_encoder,
                        device=device,
                    )
                    best_q_eval = select_best_query(
                        out_eval,
                        sample["target_mask_resized"],
                        sample["target_box"],
                        args.resolution,
                    )
                    pred_eval = out_eval["pred_masks"][0, best_q_eval]
                    ph, pw = pred_eval.shape
                    tgt = F.interpolate(
                        sample["target_mask_resized"].unsqueeze(0).unsqueeze(0),
                        size=(ph, pw), mode="nearest",
                    ).squeeze(0).squeeze(0)
                    all_ious.append(compute_iou(pred_eval, tgt))
            mean_iou = np.mean(all_ious)
            min_iou = min(all_ious)
            max_iou = max(all_ious)
        else:
            mean_iou = np.mean([step_ious[i] for i in batch_indices]) if batch_indices else 0.0
            all_ious = step_ious
            min_iou = min(step_ious[i] for i in batch_indices) if batch_indices else 0.0
            max_iou = max(step_ious[i] for i in batch_indices) if batch_indices else 0.0

        loss_val = total_loss.item() if not torch.isnan(total_loss) else float("inf")
        history["step"].append(step)
        history["loss"].append(loss_val)
        history["mean_iou"].append(float(mean_iou))
        history["per_image_iou"].append([float(x) for x in all_ious])
        history["mask_loss"].append(step_mask_loss / max(valid_count, 1))
        history["score_loss"].append(step_score_loss / max(valid_count, 1))
        history["box_loss"].append(step_box_loss / max(valid_count, 1))

        if mean_iou > best_mean_iou:
            best_mean_iou = float(mean_iou)
            best_embedding = opt_embedding.detach().clone()

        if step % args.log_interval == 0 or step == args.num_steps - 1:
            tokens = decode_nearest_tokens(opt_embedding, text_encoder)
            print(
                f"  Step {step:4d} | Loss: {loss_val:.4f} | "
                f"Mean IoU: {mean_iou:.4f} (min {min_iou:.4f}, max {max_iou:.4f}) | "
                f"Tokens: {tokens[:50]}"
            )

    # ---- Save results ----
    print(f"\nDone! Best mean IoU: {best_mean_iou:.4f}")

    if best_embedding is None:
        best_embedding = opt_embedding.detach().clone()

    emb_path = os.path.join(args.output_dir, "class_embedding.pt")
    torch.save(
        {
            "embedding": best_embedding.cpu(),
            "context_length": ctx_len,
            "width": width,
            "best_mean_iou": best_mean_iou,
            "seed_text": args.seed_text,
            "num_steps": args.num_steps,
            "num_images": N,
        },
        emb_path,
    )
    print(f"  Saved class embedding to: {emb_path}")

    hist_path = os.path.join(args.output_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Saved history to: {hist_path}")

    final_tokens = decode_nearest_tokens(best_embedding, text_encoder)
    tokens_path = os.path.join(args.output_dir, "nearest_tokens.txt")
    with open(tokens_path, "w") as f:
        f.write(final_tokens)
    print(f"  Nearest tokens: {final_tokens}")

    # Per-image final IoU breakdown
    print(f"\n  Per-image final IoU:")
    final_ious = history["per_image_iou"][-1] if history["per_image_iou"] else []
    for i, (img_path, _) in enumerate(pairs):
        iou_val = final_ious[i] if i < len(final_ious) else 0.0
        print(f"    [{i}] {Path(img_path).name}: {iou_val:.4f}")


if __name__ == "__main__":
    args = parse_args()
    optimize_class(args)
