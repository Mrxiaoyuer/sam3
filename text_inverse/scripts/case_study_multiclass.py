#!/usr/bin/env python3
"""
Multi-class rare object detection case study.

Learns a single class embedding per class across the FULL dataset,
then evaluates generalization on held-out images.

Classes:
  1. CAMO       — camouflaged animals/objects (1250 images)
  2. Kvasir-SEG — gastrointestinal polyps     (1000 images)

For each class we:
  1. Download the full dataset
  2. Split 80/20 into train/val
  3. Train a single shared embedding on the train split
  4. Evaluate on the val split (never seen during optimization)
  5. Report convergence and generalization metrics
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
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


def download_camo(out_dir: str, max_samples: int = 0) -> Tuple[List[str], List[str]]:
    """Download CAMO dataset, return (image_paths, mask_paths)."""
    from datasets import load_dataset
    img_dir = os.path.join(out_dir, "camo", "images")
    mask_dir = os.path.join(out_dir, "camo", "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    ds = load_dataset("PassbyGrocer/CAMO", split="train", streaming=True)
    img_paths, mask_paths = [], []
    for i, item in enumerate(ds):
        if max_samples > 0 and i >= max_samples:
            break
        fname = f"camo_{i:04d}.png"
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        if not os.path.exists(img_path):
            item["image"].convert("RGB").save(img_path)
            gt = np.array(item["gt"].convert("L"))
            gt_bin = ((gt > 128).astype(np.uint8)) * 255
            Image.fromarray(gt_bin).save(mask_path)
        img_paths.append(img_path)
        mask_paths.append(mask_path)
        if (i + 1) % 100 == 0:
            print(f"    CAMO: {i+1} images downloaded...")
    return img_paths, mask_paths


def download_kvasir(out_dir: str, max_samples: int = 0) -> Tuple[List[str], List[str]]:
    """Download Kvasir-SEG dataset, return (image_paths, mask_paths)."""
    from datasets import load_dataset
    img_dir = os.path.join(out_dir, "kvasir", "images")
    mask_dir = os.path.join(out_dir, "kvasir", "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    ds = load_dataset("kowndinya23/Kvasir-SEG", split="train", streaming=True)
    img_paths, mask_paths = [], []
    for i, item in enumerate(ds):
        if max_samples > 0 and i >= max_samples:
            break
        fname = f"kvasir_{i:04d}.png"
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        if not os.path.exists(img_path):
            item["image"].convert("RGB").save(img_path)
            gt = np.array(item["annotation"].convert("L"))
            gt_bin = ((gt > 128).astype(np.uint8)) * 255
            Image.fromarray(gt_bin).save(mask_path)
        img_paths.append(img_path)
        mask_paths.append(mask_path)
        if (i + 1) % 100 == 0:
            print(f"    Kvasir: {i+1} images downloaded...")
    return img_paths, mask_paths


def precompute_sample(model, img_path, mask_path, resolution, device):
    """Pre-compute backbone features for a single image."""
    img_tensor = load_and_preprocess_image(img_path, resolution, device)
    with torch.no_grad():
        backbone_out = model.backbone.forward_image(img_tensor)
    target_mask_full, target_mask_resized, orig_h, orig_w = load_target_mask(
        mask_path, resolution, device
    )
    target_box = mask_to_bbox_cxcywh(target_mask_resized)
    return {
        "backbone_out": backbone_out,
        "target_mask_resized": target_mask_resized,
        "target_box": target_box,
    }


def train_class_embedding(
    model,
    train_samples: List[Dict],
    text_encoder,
    device: torch.device,
    seed_text: str = None,
    num_steps: int = 300,
    lr: float = 0.015,
    batch_size: int = 8,
    ctx_len: int = 16,
    log_interval: int = 25,
    resolution: int = 1008,
    class_name: str = "unknown",
) -> Tuple[torch.Tensor, Dict]:
    """Train a single class embedding across multiple images."""

    width = text_encoder.encoder.width

    if seed_text:
        opt_embedding = init_embedding_from_text(seed_text, text_encoder, ctx_len, device)
    else:
        opt_embedding = init_embedding_random(ctx_len, width, device)
    opt_embedding = torch.nn.Parameter(opt_embedding)
    optimizer = torch.optim.Adam([opt_embedding], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=lr * 0.01
    )

    N = len(train_samples)
    history = {"step": [], "loss": [], "mean_iou": []}
    best_mean_iou = 0.0
    best_embedding = None

    for step in range(num_steps):
        optimizer.zero_grad()

        # Mini-batch
        bs = min(batch_size, N)
        indices = np.random.choice(N, size=bs, replace=False).tolist()

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        step_ious = []
        valid = 0

        for idx in indices:
            s = train_samples[idx]
            out = forward_with_embedding(
                model=model,
                backbone_out=s["backbone_out"],
                opt_embedding=opt_embedding,
                text_encoder=text_encoder,
                device=device,
            )
            with torch.no_grad():
                best_q = select_best_query(
                    out, s["target_mask_resized"], s["target_box"], resolution
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

            ph, pw = pred_mask.shape
            tgt = F.interpolate(
                s["target_mask_resized"].unsqueeze(0).unsqueeze(0),
                size=(ph, pw), mode="nearest"
            ).squeeze(0).squeeze(0)

            l_mask = mask_loss(pred_mask, tgt)
            l_score = F.binary_cross_entropy_with_logits(
                pred_logit, torch.ones_like(pred_logit)
            )
            l_box = F.l1_loss(pred_box, s["target_box"]) + box_giou_loss(
                pred_box, s["target_box"]
            )
            img_loss = 5.0 * l_mask + 2.0 * l_score + 1.0 * l_box

            if not (torch.isnan(img_loss) or torch.isinf(img_loss)):
                total_loss = total_loss + img_loss / bs
                valid += 1

            with torch.no_grad():
                step_ious.append(compute_iou(pred_mask, tgt))

        if valid > 0 and not (torch.isnan(total_loss) or torch.isinf(total_loss)):
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([opt_embedding], max_norm=1.0)
            optimizer.step()
        else:
            optimizer.zero_grad()
        scheduler.step()

        with torch.no_grad():
            nm = torch.isnan(opt_embedding)
            if nm.any():
                opt_embedding.data[nm] = 0.0

        mean_iou = float(np.mean(step_ious)) if step_ious else 0.0
        loss_val = total_loss.item() if not torch.isnan(total_loss) else float("inf")
        history["step"].append(step)
        history["loss"].append(loss_val)
        history["mean_iou"].append(mean_iou)

        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            best_embedding = opt_embedding.detach().clone()

        if step % log_interval == 0 or step == num_steps - 1:
            tokens = decode_nearest_tokens(opt_embedding, text_encoder)
            print(
                f"    Step {step:4d} | Loss: {loss_val:.4f} | "
                f"Train IoU: {mean_iou:.4f} | Tokens: {tokens[:50]}"
            )

    if best_embedding is None:
        best_embedding = opt_embedding.detach().clone()

    return best_embedding, history


def evaluate_embedding(
    model,
    embedding: torch.Tensor,
    samples: List[Dict],
    text_encoder,
    device: torch.device,
    resolution: int = 1008,
) -> List[float]:
    """Evaluate a class embedding on a set of images. Returns per-image IoUs."""
    ious = []
    with torch.no_grad():
        for s in samples:
            out = forward_with_embedding(
                model=model,
                backbone_out=s["backbone_out"],
                opt_embedding=embedding,
                text_encoder=text_encoder,
                device=device,
            )
            best_q = select_best_query(
                out, s["target_mask_resized"], s["target_box"], resolution
            )
            pred_mask = out["pred_masks"][0, best_q]
            ph, pw = pred_mask.shape
            tgt = F.interpolate(
                s["target_mask_resized"].unsqueeze(0).unsqueeze(0),
                size=(ph, pw), mode="nearest"
            ).squeeze(0).squeeze(0)
            ious.append(compute_iou(pred_mask, tgt))
    return ious


def main():
    parser = argparse.ArgumentParser(description="Multi-class rare object case study")
    parser.add_argument("--data-dir", default="outputs/case_study_data")
    parser.add_argument("--output-dir", default="outputs/case_study_results")
    parser.add_argument("--max-train", type=int, default=50,
                        help="Max training images per class (0=all)")
    parser.add_argument("--max-val", type=int, default=20,
                        help="Max val images per class")
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ==============================================================
    #  1. Download datasets
    # ==============================================================
    total_samples = args.max_train + args.max_val

    classes = {
        "camo": {
            "download": download_camo,
            "seed_text": "camouflaged animal hidden",
            "description": "Camouflaged objects (CAMO)",
        },
        "polyp": {
            "download": download_kvasir,
            "seed_text": "gastrointestinal polyp tissue",
            "description": "Gastrointestinal polyps (Kvasir-SEG)",
        },
    }

    all_data = {}
    for cls_name, cls_info in classes.items():
        print(f"\n{'=' * 70}")
        print(f"Downloading: {cls_info['description']}")
        print(f"{'=' * 70}")
        img_paths, mask_paths = cls_info["download"](
            args.data_dir, max_samples=total_samples
        )
        print(f"  Downloaded {len(img_paths)} images")

        # Split train/val
        n_total = len(img_paths)
        n_train = min(args.max_train, int(n_total * 0.8))
        n_val = min(args.max_val, n_total - n_train)

        # Reproducible shuffle
        rng = np.random.RandomState(42)
        indices = rng.permutation(n_total)
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]

        all_data[cls_name] = {
            "train_imgs": [img_paths[i] for i in train_idx],
            "train_masks": [mask_paths[i] for i in train_idx],
            "val_imgs": [img_paths[i] for i in val_idx],
            "val_masks": [mask_paths[i] for i in val_idx],
            "seed_text": cls_info["seed_text"],
            "description": cls_info["description"],
        }
        print(f"  Train: {len(train_idx)} images, Val: {len(val_idx)} images")

    # ==============================================================
    #  2. Load model
    # ==============================================================
    print(f"\n{'=' * 70}")
    print("Loading SAM3 model...")
    print(f"{'=' * 70}")
    from sam3 import build_sam3_image_model

    model = build_sam3_image_model(
        device=str(device), eval_mode=True,
        checkpoint_path=args.checkpoint_path, enable_segmentation=True,
    )
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    text_encoder = model.backbone.language_backbone
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # ==============================================================
    #  3. Per-class: precompute features, train, evaluate
    # ==============================================================
    results = {}

    for cls_name, data in all_data.items():
        info = classes[cls_name]
        print(f"\n{'#' * 70}")
        print(f"#  CLASS: {info['description']}")
        print(f"#  Train: {len(data['train_imgs'])} images, Val: {len(data['val_imgs'])} images")
        print(f"#  Seed: '{data['seed_text']}'")
        print(f"{'#' * 70}")

        # Pre-compute train features
        print(f"\n  Pre-computing train features...")
        t0 = time.time()
        train_samples = []
        for img_p, mask_p in zip(data["train_imgs"], data["train_masks"]):
            train_samples.append(
                precompute_sample(model, img_p, mask_p, args.resolution, device)
            )
        print(f"  Train features: {len(train_samples)} in {time.time()-t0:.1f}s")

        # Pre-compute val features
        print(f"  Pre-computing val features...")
        t0 = time.time()
        val_samples = []
        for img_p, mask_p in zip(data["val_imgs"], data["val_masks"]):
            val_samples.append(
                precompute_sample(model, img_p, mask_p, args.resolution, device)
            )
        print(f"  Val features: {len(val_samples)} in {time.time()-t0:.1f}s")

        # Train class embedding
        print(f"\n  Training class embedding ({args.num_steps} steps, batch={args.batch_size})...")
        embedding, history = train_class_embedding(
            model=model,
            train_samples=train_samples,
            text_encoder=text_encoder,
            device=device,
            seed_text=data["seed_text"],
            num_steps=args.num_steps,
            lr=args.lr,
            batch_size=args.batch_size,
            ctx_len=16,
            log_interval=25,
            resolution=args.resolution,
            class_name=cls_name,
        )

        # Evaluate on TRAIN set
        print(f"\n  Evaluating on TRAIN set ({len(train_samples)} images)...")
        train_ious = evaluate_embedding(
            model, embedding, train_samples, text_encoder, device, args.resolution
        )
        train_mean = float(np.mean(train_ious))
        train_std = float(np.std(train_ious))

        # Evaluate on VAL set (UNSEEN images)
        print(f"  Evaluating on VAL set ({len(val_samples)} images, UNSEEN)...")
        val_ious = evaluate_embedding(
            model, embedding, val_samples, text_encoder, device, args.resolution
        )
        val_mean = float(np.mean(val_ious))
        val_std = float(np.std(val_ious))

        tokens = decode_nearest_tokens(embedding, text_encoder)

        # Cross-class evaluation: test this embedding on other classes' val sets
        cross_ious = {}
        for other_cls, other_data in all_data.items():
            if other_cls == cls_name:
                continue
            other_val = []
            for img_p, mask_p in zip(
                other_data["val_imgs"][:10], other_data["val_masks"][:10]
            ):
                other_val.append(
                    precompute_sample(model, img_p, mask_p, args.resolution, device)
                )
            other_ious = evaluate_embedding(
                model, embedding, other_val, text_encoder, device, args.resolution
            )
            cross_ious[other_cls] = float(np.mean(other_ious))

        results[cls_name] = {
            "description": info["description"],
            "seed_text": data["seed_text"],
            "n_train": len(train_samples),
            "n_val": len(val_samples),
            "train_iou_mean": train_mean,
            "train_iou_std": train_std,
            "val_iou_mean": val_mean,
            "val_iou_std": val_std,
            "val_ious": val_ious,
            "generalization_gap": train_mean - val_mean,
            "cross_class_ious": cross_ious,
            "final_tokens": tokens[:80],
            "history_loss_start": history["loss"][0] if history["loss"] else 0,
            "history_loss_end": history["loss"][-1] if history["loss"] else 0,
        }

        # Save per-class embedding
        cls_dir = os.path.join(args.output_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        torch.save(embedding.cpu(), os.path.join(cls_dir, "class_embedding.pt"))
        with open(os.path.join(cls_dir, "history.json"), "w") as f:
            json.dump(history, f)

        print(f"\n  {'─' * 60}")
        print(f"  RESULTS: {info['description']}")
        print(f"  {'─' * 60}")
        print(f"  Train IoU: {train_mean:.4f} ± {train_std:.4f}")
        print(f"  Val   IoU: {val_mean:.4f} ± {val_std:.4f} (UNSEEN)")
        print(f"  Gen. gap:  {train_mean - val_mean:+.4f}")
        print(f"  Loss:      {results[cls_name]['history_loss_start']:.2f} → {results[cls_name]['history_loss_end']:.2f}")
        print(f"  Tokens:    {tokens[:60]}")
        for other_cls, ciou in cross_ious.items():
            print(f"  Cross({other_cls}): {ciou:.4f}")

    # ==============================================================
    #  4. Final summary
    # ==============================================================
    print(f"\n{'=' * 70}")
    print("MULTI-CLASS CASE STUDY SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Class':<25} {'Train IoU':>10} {'Val IoU':>10} {'Gap':>8} {'Converged':>10}")
    print("-" * 70)
    for cls_name, r in results.items():
        converged = "YES" if r["val_iou_mean"] > 0.3 else "no"
        print(
            f"{r['description']:<25} "
            f"{r['train_iou_mean']:>10.4f} "
            f"{r['val_iou_mean']:>10.4f} "
            f"{r['generalization_gap']:>+8.4f} "
            f"{converged:>10}"
        )

    # Cross-class confusion
    print(f"\nCross-class IoU (embedding from row, evaluated on column):")
    cls_names = list(results.keys())
    header = f"{'':>12}" + "".join(f"{c:>12}" for c in cls_names)
    print(header)
    for cls_name in cls_names:
        row = f"{cls_name:>12}"
        for other in cls_names:
            if other == cls_name:
                row += f"{results[cls_name]['val_iou_mean']:>12.4f}"
            elif other in results[cls_name]["cross_class_ious"]:
                row += f"{results[cls_name]['cross_class_ious'][other]:>12.4f}"
            else:
                row += f"{'N/A':>12}"
        print(row)

    print(f"\nKey: diagonal = same-class val IoU, off-diagonal = cross-class IoU")
    print(f"Good class embeddings should have HIGH diagonal, LOW off-diagonal.")

    # Save summary
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
