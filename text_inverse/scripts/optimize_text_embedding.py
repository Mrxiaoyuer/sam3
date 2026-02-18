# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
Gradient ascent optimization to find token embeddings that fire on specific masks.

Given a frozen SAM3 model, a target image, and a target binary mask (or bbox),
this script optimizes continuous token embeddings in the TextTransformer's
embedding space to produce the desired mask output.

Usage:
    python scripts/optimize_text_embedding.py \
        --image path/to/image.jpg \
        --target-mask path/to/binary_mask.png \
        --seed-text "building" \
        --num-steps 200 \
        --lr 0.01 \
        --output-dir outputs/optimization_run
"""

import argparse
import json
import os
import sys
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
        description="Optimize token embeddings for target mask via gradient ascent"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to input image"
    )
    parser.add_argument(
        "--target-mask",
        type=str,
        required=True,
        help="Path to target binary mask (white=foreground, black=background)",
    )
    parser.add_argument(
        "--seed-text",
        type=str,
        default=None,
        help="Optional seed text to initialize embeddings from (e.g. 'damaged building')",
    )
    parser.add_argument(
        "--num-steps", type=int, default=200, help="Number of optimization steps"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/optimization_run",
        help="Directory to save results",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=16,
        help="Number of token positions to optimize (max 32)",
    )
    parser.add_argument(
        "--lambda-mask", type=float, default=5.0, help="Weight for mask loss"
    )
    parser.add_argument(
        "--lambda-score", type=float, default=2.0, help="Weight for score loss"
    )
    parser.add_argument(
        "--lambda-box", type=float, default=1.0, help="Weight for box loss"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (auto-detected if not set)",
    )
    parser.add_argument(
        "--resolution", type=int, default=1008, help="Image resolution for SAM3"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to SAM3 checkpoint (downloads from HF if not set)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--save-progression",
        action="store_true",
        help="Save mask predictions at each log interval",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Dice loss between predicted and target masks.

    Args:
        pred: Predicted mask logits [H, W] (pre-sigmoid)
        target: Target binary mask [H, W] (0 or 1)
    """
    pred_prob = pred.sigmoid().flatten()
    target_flat = target.flatten()
    intersection = (pred_prob * target_flat).sum()
    return 1.0 - (2.0 * intersection + 1.0) / (
        pred_prob.sum() + target_flat.sum() + 1.0
    )


def mask_loss(
    pred_mask: torch.Tensor, target_mask: torch.Tensor
) -> torch.Tensor:
    """Combined BCE + Dice mask loss.

    Args:
        pred_mask: Predicted mask logits [H, W] (pre-sigmoid)
        target_mask: Target binary mask [H, W] (0 or 1)
    """
    bce = F.binary_cross_entropy_with_logits(pred_mask, target_mask)
    dice = dice_loss(pred_mask, target_mask)
    return bce + dice


def box_giou_loss(
    pred_box: torch.Tensor, target_box: torch.Tensor
) -> torch.Tensor:
    """Generalized IoU loss for boxes in [cx, cy, w, h] format (normalized).

    Args:
        pred_box: Predicted box [4] in cxcywh format
        target_box: Target box [4] in cxcywh format
    """
    # Convert to xyxy
    def cxcywh_to_xyxy(box):
        cx, cy, w, h = box.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

    pred_xyxy = cxcywh_to_xyxy(pred_box)
    tgt_xyxy = cxcywh_to_xyxy(target_box)

    # Intersection
    inter_tl = torch.max(pred_xyxy[:2], tgt_xyxy[:2])
    inter_br = torch.min(pred_xyxy[2:], tgt_xyxy[2:])
    inter_wh = (inter_br - inter_tl).clamp(min=0)
    inter_area = inter_wh[0] * inter_wh[1]

    # Union
    pred_area = (pred_xyxy[2] - pred_xyxy[0]) * (pred_xyxy[3] - pred_xyxy[1])
    tgt_area = (tgt_xyxy[2] - tgt_xyxy[0]) * (tgt_xyxy[3] - tgt_xyxy[1])
    union_area = pred_area + tgt_area - inter_area

    iou = inter_area / (union_area + 1e-6)

    # Enclosing box
    encl_tl = torch.min(pred_xyxy[:2], tgt_xyxy[:2])
    encl_br = torch.max(pred_xyxy[2:], tgt_xyxy[2:])
    encl_area = (encl_br[0] - encl_tl[0]) * (encl_br[1] - encl_tl[1])

    giou = iou - (encl_area - union_area) / (encl_area + 1e-6)
    return 1.0 - giou


def compute_iou(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> float:
    """Compute IoU between binary masks."""
    pred_bin = (pred_mask.sigmoid() > 0.5).float()
    intersection = (pred_bin * target_mask).sum()
    union = pred_bin.sum() + target_mask.sum() - intersection
    return (intersection / (union + 1e-6)).item()


# ---------------------------------------------------------------------------
# Target mask utilities
# ---------------------------------------------------------------------------


def load_target_mask(
    mask_path: str, resolution: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Load and preprocess a target binary mask.

    Returns:
        target_mask_full: Binary mask at original resolution [H, W]
        target_mask_resized: Binary mask resized to model resolution [H_model, W_model]
        orig_h: Original image height
        orig_w: Original image width
    """
    mask_img = Image.open(mask_path).convert("L")
    orig_w, orig_h = mask_img.size

    # Full resolution mask
    mask_np = np.array(mask_img).astype(np.float32) / 255.0
    mask_np = (mask_np > 0.5).astype(np.float32)
    target_mask_full = torch.from_numpy(mask_np).to(device)

    # Resized mask for loss computation during optimization
    target_mask_resized = (
        F.interpolate(
            target_mask_full.unsqueeze(0).unsqueeze(0),
            size=(resolution, resolution),
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
    )

    return target_mask_full, target_mask_resized, orig_h, orig_w


def mask_to_bbox_cxcywh(mask: torch.Tensor) -> torch.Tensor:
    """Compute normalized [cx, cy, w, h] bounding box from a binary mask."""
    h, w = mask.shape
    ys, xs = torch.where(mask > 0.5)
    if len(ys) == 0:
        return torch.tensor([0.5, 0.5, 1.0, 1.0], device=mask.device)
    x0, x1 = xs.min().float(), xs.max().float()
    y0, y1 = ys.min().float(), ys.max().float()
    cx = (x0 + x1) / 2.0 / w
    cy = (y0 + y1) / 2.0 / h
    bw = (x1 - x0 + 1) / w
    bh = (y1 - y0 + 1) / h
    return torch.tensor([cx, cy, bw, bh], device=mask.device)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def load_and_preprocess_image(
    image_path: str, resolution: int, device: torch.device
) -> torch.Tensor:
    """Load an image and preprocess it for SAM3."""
    img = Image.open(image_path).convert("RGB")
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(resolution, max_size=resolution + 1),
            v2.CenterCrop(resolution),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor


# ---------------------------------------------------------------------------
# Embedding initialization
# ---------------------------------------------------------------------------


def init_embedding_from_text(
    text: str,
    text_encoder,
    context_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Initialize learnable embedding from a seed text prompt.

    Returns:
        Tensor of shape [1, context_length, width] (width=1024 for SAM3)
    """
    tokenizer = text_encoder.tokenizer
    token_ids = tokenizer(text, context_length=context_length).to(device)
    with torch.no_grad():
        embedding = text_encoder.encoder.token_embedding(token_ids)
    return embedding.clone()  # [1, context_length, 1024]


def init_embedding_random(
    context_length: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Initialize learnable embedding with random normal values.

    Returns:
        Tensor of shape [1, context_length, width]
    """
    return torch.randn(1, context_length, width, device=device) * 0.02


# ---------------------------------------------------------------------------
# Modified forward pass
# ---------------------------------------------------------------------------


def forward_with_embedding(
    model,
    backbone_out: Dict,
    opt_embedding: torch.Tensor,
    text_encoder,
    device: torch.device,
) -> Dict:
    """Run the SAM3 forward pass using the optimized embedding instead of text.

    This bypasses VETextEncoder.forward() and manually threads the embedding
    through the TextTransformer, then runs the full detection/segmentation
    pipeline.

    Args:
        model: The Sam3Image model
        backbone_out: Pre-computed backbone output (image features)
        opt_embedding: Learnable embedding [1, seq_len, 1024]
        text_encoder: VETextEncoder (model.backbone.language_backbone)
        device: torch device

    Returns:
        Model output dict with pred_masks, pred_logits, pred_boxes, etc.
    """
    seq_len = opt_embedding.shape[1]
    encoder = text_encoder.encoder  # TextTransformer

    # Step 1: Run TextTransformer with our learnable embedding
    x = opt_embedding + encoder.positional_embedding[:seq_len]

    attn_mask = encoder.attn_mask
    if attn_mask is not None:
        attn_mask = attn_mask[:seq_len, :seq_len]

    x = encoder.transformer(x, attn_mask=attn_mask)
    x = encoder.ln_final(x)

    # Step 2: Resize to d_model (1024 → 256)
    text_memory = x.transpose(0, 1)  # [seq_len, batch, 1024]
    text_memory_resized = text_encoder.resizer(text_memory)  # [seq_len, batch, 256]

    # Step 3: Build text attention mask (all tokens valid)
    text_attention_mask = torch.zeros(
        1, seq_len, dtype=torch.bool, device=device
    )

    # Step 4: Inject into backbone_out
    backbone_out_copy = dict(backbone_out)
    backbone_out_copy["language_features"] = text_memory_resized
    backbone_out_copy["language_mask"] = text_attention_mask
    backbone_out_copy["language_embeds"] = opt_embedding.transpose(0, 1)

    # Step 5: Run forward_grounding with dummy geometric prompt
    from sam3.model.data_misc import FindStage

    find_stage = FindStage(
        img_ids=torch.tensor([0], device=device, dtype=torch.long),
        text_ids=torch.tensor([0], device=device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )
    geometric_prompt = model._get_dummy_prompt()

    out = model.forward_grounding(
        backbone_out=backbone_out_copy,
        find_input=find_stage,
        geometric_prompt=geometric_prompt,
        find_target=None,
    )
    return out


# ---------------------------------------------------------------------------
# Query matching
# ---------------------------------------------------------------------------


def select_best_query(
    out: Dict,
    target_mask: torch.Tensor,
    target_box: torch.Tensor,
    model_resolution: int,
) -> int:
    """Select the query whose prediction best matches the target.

    Strategy: Pick query with highest mask IoU. If none > 0.05, use box IoU.
    """
    pred_masks = out["pred_masks"]  # [batch, num_queries, H, W]
    pred_boxes = out["pred_boxes"]  # [batch, num_queries, 4] (eval) or [layers, batch, queries, 4]

    # Resize target mask to prediction resolution
    pred_h, pred_w = pred_masks.shape[-2:]
    target_resized = F.interpolate(
        target_mask.unsqueeze(0).unsqueeze(0),
        size=(pred_h, pred_w),
        mode="nearest",
    ).squeeze(0).squeeze(0)

    num_queries = pred_masks.shape[1]
    best_iou = -1.0
    best_idx = 0

    for q in range(num_queries):
        iou = compute_iou(pred_masks[0, q], target_resized)
        if iou > best_iou:
            best_iou = iou
            best_idx = q

    return best_idx


# ---------------------------------------------------------------------------
# Nearest-token decode
# ---------------------------------------------------------------------------


def decode_nearest_tokens(
    opt_embedding: torch.Tensor,
    text_encoder,
) -> str:
    """Find the nearest discrete tokens for each position in the optimized embedding.

    Returns decoded string of nearest tokens.
    """
    with torch.no_grad():
        vocab_embeddings = text_encoder.encoder.token_embedding.weight  # [V, 1024]
        opt_flat = opt_embedding.squeeze(0)  # [seq_len, 1024]

        # Compute cosine similarity
        opt_norm = F.normalize(opt_flat, dim=-1)
        vocab_norm = F.normalize(vocab_embeddings, dim=-1)
        sim = torch.matmul(opt_norm, vocab_norm.T)  # [seq_len, V]
        nearest_ids = sim.argmax(dim=-1)  # [seq_len]

        tokenizer = text_encoder.tokenizer
        tokens = tokenizer.decode(nearest_ids.cpu().tolist())

    return tokens


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------


def optimize(args):
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load model ----
    print("Loading SAM3 model...")
    from sam3 import build_sam3_image_model

    model = build_sam3_image_model(
        device=str(device),
        eval_mode=True,
        checkpoint_path=args.checkpoint_path,
        enable_segmentation=True,
    )

    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    text_encoder = model.backbone.language_backbone

    # ---- Load image ----
    print(f"Loading image: {args.image}")
    img_tensor = load_and_preprocess_image(args.image, args.resolution, device)

    # Pre-compute image features (frozen, done once)
    print("Computing image features...")
    with torch.no_grad():
        backbone_out = model.backbone.forward_image(img_tensor)

    # ---- Load target mask ----
    print(f"Loading target mask: {args.target_mask}")
    target_mask_full, target_mask_resized, orig_h, orig_w = load_target_mask(
        args.target_mask, args.resolution, device
    )
    target_box = mask_to_bbox_cxcywh(target_mask_resized)
    print(f"  Target bbox (cxcywh): {target_box.cpu().tolist()}")

    # ---- Initialize learnable embedding ----
    width = text_encoder.encoder.width  # 1024
    ctx_len = min(args.context_length, text_encoder.context_length)

    if args.seed_text:
        print(f"Initializing from seed text: '{args.seed_text}'")
        opt_embedding = init_embedding_from_text(
            args.seed_text, text_encoder, ctx_len, device
        )
    else:
        print("Initializing with random embeddings")
        opt_embedding = init_embedding_random(ctx_len, width, device)

    opt_embedding = nn.Parameter(opt_embedding)  # [1, ctx_len, 1024]

    # ---- Setup optimizer ----
    optimizer = torch.optim.Adam([opt_embedding], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps, eta_min=args.lr * 0.01
    )

    # ---- Optimization loop ----
    print(f"\nStarting optimization for {args.num_steps} steps...")
    history = {"step": [], "loss": [], "iou": [], "mask_loss": [], "score_loss": [], "box_loss": []}
    best_iou = 0.0
    best_embedding = None

    for step in range(args.num_steps):
        optimizer.zero_grad()

        # Forward pass (no inference_mode!)
        out = forward_with_embedding(
            model=model,
            backbone_out=backbone_out,
            opt_embedding=opt_embedding,
            text_encoder=text_encoder,
            device=device,
        )

        # Select best matching query
        with torch.no_grad():
            best_q = select_best_query(
                out, target_mask_resized, target_box, args.resolution
            )

        # Get predictions for the best query
        # In eval mode, SAM3 collapses the layer dim:
        #   pred_logits: [batch, num_queries, 1]
        #   pred_boxes:  [batch, num_queries, 4]
        #   pred_masks:  [batch, num_queries, H, W]
        pred_masks = out["pred_masks"]
        pred_logits = out["pred_logits"]
        pred_boxes = out["pred_boxes"]

        pred_mask_q = pred_masks[0, best_q]  # [H, W]

        # Handle both eval (3D) and train (4D) shapes for logits/boxes
        if pred_logits.dim() == 3:
            # eval mode: [batch, queries, 1]
            pred_logit_q = pred_logits[0, best_q, 0]
            pred_box_q = pred_boxes[0, best_q]
        else:
            # train mode: [layers, batch, queries, 1]
            pred_logit_q = pred_logits[-1, 0, best_q, 0]
            pred_box_q = pred_boxes[-1, 0, best_q]

        # Resize target mask to prediction mask size
        pred_h, pred_w = pred_mask_q.shape
        target_for_loss = F.interpolate(
            target_mask_resized.unsqueeze(0).unsqueeze(0),
            size=(pred_h, pred_w),
            mode="nearest",
        ).squeeze(0).squeeze(0)

        # Compute losses
        l_mask = mask_loss(pred_mask_q, target_for_loss)
        l_score = F.binary_cross_entropy_with_logits(
            pred_logit_q,
            torch.ones_like(pred_logit_q),
        )
        l_box = (
            F.l1_loss(pred_box_q, target_box)
            + box_giou_loss(pred_box_q, target_box)
        )

        # Presence loss if available
        # In eval mode, presence_logit_dec is [batch, 1] (per-image, not per-query)
        l_presence = torch.tensor(0.0, device=device)
        if "presence_logit_dec" in out:
            presence = out["presence_logit_dec"]
            if presence is not None and presence.numel() > 0:
                # Squeeze to scalar and maximize it
                pres_logit = presence.view(-1)[0]
                l_presence = F.binary_cross_entropy_with_logits(
                    pres_logit,
                    torch.ones_like(pres_logit),
                )

        total_loss = (
            args.lambda_mask * l_mask
            + args.lambda_score * (l_score + l_presence)
            + args.lambda_box * l_box
        )

        # Backward + step (with NaN guard and gradient clipping)
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            # Skip this step if loss is NaN/Inf (common with random model weights)
            optimizer.zero_grad()
        else:
            total_loss.backward()
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_([opt_embedding], max_norm=1.0)
            optimizer.step()
        scheduler.step()

        # Replace any NaN in embedding with zeros
        with torch.no_grad():
            nan_mask = torch.isnan(opt_embedding)
            if nan_mask.any():
                opt_embedding.data[nan_mask] = 0.0

        # Track metrics
        with torch.no_grad():
            cur_iou = compute_iou(pred_mask_q, target_for_loss)

        loss_val = total_loss.item() if not torch.isnan(total_loss) else float('inf')
        history["step"].append(step)
        history["loss"].append(loss_val)
        history["iou"].append(cur_iou)
        history["mask_loss"].append(l_mask.item() if not torch.isnan(l_mask) else float('inf'))
        history["score_loss"].append(l_score.item() if not torch.isnan(l_score) else float('inf'))
        history["box_loss"].append(l_box.item() if not torch.isnan(l_box) else float('inf'))

        if cur_iou > best_iou:
            best_iou = cur_iou
            best_embedding = opt_embedding.detach().clone()

        # Logging
        if step % args.log_interval == 0 or step == args.num_steps - 1:
            score = pred_logit_q.sigmoid().item()
            tokens_str = decode_nearest_tokens(opt_embedding, text_encoder)
            print(
                f"  Step {step:4d} | Loss: {total_loss.item():.4f} | "
                f"IoU: {cur_iou:.4f} | Score: {score:.4f} | "
                f"Mask: {l_mask.item():.4f} | Box: {l_box.item():.4f} | "
                f"Nearest tokens: {tokens_str[:60]}"
            )

            if args.save_progression:
                save_mask_comparison(
                    pred_mask_q.detach(),
                    target_for_loss,
                    os.path.join(args.output_dir, f"mask_step_{step:04d}.png"),
                )

    # ---- Save results ----
    print(f"\nOptimization complete! Best IoU: {best_iou:.4f}")

    # Save optimized embedding
    emb_path = os.path.join(args.output_dir, "optimized_embedding.pt")
    torch.save(
        {
            "embedding": best_embedding.cpu(),
            "context_length": ctx_len,
            "width": width,
            "best_iou": best_iou,
            "seed_text": args.seed_text,
            "num_steps": args.num_steps,
        },
        emb_path,
    )
    print(f"  Saved optimized embedding to: {emb_path}")

    # Save loss history
    hist_path = os.path.join(args.output_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Saved history to: {hist_path}")

    # Save nearest tokens
    final_tokens = decode_nearest_tokens(best_embedding, text_encoder)
    tokens_path = os.path.join(args.output_dir, "nearest_tokens.txt")
    with open(tokens_path, "w") as f:
        f.write(final_tokens)
    print(f"  Nearest tokens: {final_tokens}")
    print(f"  Saved to: {tokens_path}")

    # Save loss curve
    try:
        save_loss_curve(history, os.path.join(args.output_dir, "loss_curve.png"))
        print(f"  Saved loss curve to: {args.output_dir}/loss_curve.png")
    except Exception as e:
        print(f"  Warning: Could not save loss curve: {e}")

    # Save final mask comparison
    try:
        with torch.no_grad():
            out_final = forward_with_embedding(
                model=model,
                backbone_out=backbone_out,
                opt_embedding=best_embedding,
                text_encoder=text_encoder,
                device=device,
            )
            best_q = select_best_query(
                out_final, target_mask_resized, target_box, args.resolution
            )
            final_pred = out_final["pred_masks"][0, best_q]
            pred_h, pred_w = final_pred.shape
            target_cmp = F.interpolate(
                target_mask_resized.unsqueeze(0).unsqueeze(0),
                size=(pred_h, pred_w),
                mode="nearest",
            ).squeeze(0).squeeze(0)
            save_mask_comparison(
                final_pred,
                target_cmp,
                os.path.join(args.output_dir, "final_comparison.png"),
            )
            print(f"  Saved final comparison to: {args.output_dir}/final_comparison.png")
    except Exception as e:
        print(f"  Warning: Could not save final comparison: {e}")


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def save_loss_curve(history: Dict, path: str):
    """Save a plot of the loss and IoU curves."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["step"], history["loss"], label="Total Loss", color="red")
    ax1.plot(history["step"], history["mask_loss"], label="Mask Loss", alpha=0.7)
    ax1.plot(history["step"], history["score_loss"], label="Score Loss", alpha=0.7)
    ax1.plot(history["step"], history["box_loss"], label="Box Loss", alpha=0.7)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["step"], history["iou"], color="green")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("IoU")
    ax2.set_title("Mask IoU with Target")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_mask_comparison(
    pred_mask: torch.Tensor, target_mask: torch.Tensor, path: str
):
    """Save a side-by-side comparison of predicted and target masks."""
    import matplotlib.pyplot as plt

    pred_np = pred_mask.sigmoid().cpu().numpy()
    target_np = target_mask.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(target_np, cmap="gray")
    axes[0].set_title("Target Mask")
    axes[0].axis("off")

    axes[1].imshow(pred_np, cmap="gray")
    axes[1].set_title(f"Predicted (IoU: {compute_iou(pred_mask.cpu(), torch.from_numpy(target_np)):.3f})")
    axes[1].axis("off")

    # Overlay
    overlay = np.zeros((*target_np.shape, 3))
    overlay[:, :, 1] = target_np  # green = target
    overlay[:, :, 0] = (pred_np > 0.5).astype(float)  # red = pred
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Red=Pred, Green=Target)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    optimize(args)
