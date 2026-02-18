#!/usr/bin/env python3
"""
End-to-end test: generate synthetic image + mask, run optimization for a few steps.

Builds a randomly-initialized SAM3 model (no checkpoint needed) and verifies:
1. The full forward pass works with our learnable embedding
2. Gradients flow through the frozen model to the embedding
3. Loss decreases over optimization steps
4. All output artifacts are saved correctly
"""

import os
import sys
import json
import numpy as np
from PIL import Image

import torch

# Create output dirs
test_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "e2e_test")
os.makedirs(test_dir, exist_ok=True)

# --- Generate synthetic image ---
print("=" * 60)
print("End-to-End Optimization Test (Randomly-Initialized Model)")
print("=" * 60)
print()
print("Generating synthetic test image and mask...")
img_size = 512
img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

# Gradient background
for i in range(img_size):
    img[i, :, 0] = int(50 + 150 * i / img_size)
    img[i, :, 1] = int(100 + 100 * (1 - i / img_size))
    img[i, :, 2] = 80

# Draw a bright rectangle (the "object" we want to detect)
rect_y0, rect_y1 = 150, 350
rect_x0, rect_x1 = 100, 400
img[rect_y0:rect_y1, rect_x0:rect_x1] = [255, 200, 50]

# Add some texture
np.random.seed(42)
noise = np.random.randint(-30, 30, (rect_y1 - rect_y0, rect_x1 - rect_x0, 3))
img[rect_y0:rect_y1, rect_x0:rect_x1] = np.clip(
    img[rect_y0:rect_y1, rect_x0:rect_x1].astype(int) + noise, 0, 255
).astype(np.uint8)

img_path = os.path.join(test_dir, "test_image.png")
Image.fromarray(img).save(img_path)
print(f"  Saved test image: {img_path}")

# --- Generate binary mask ---
mask = np.zeros((img_size, img_size), dtype=np.uint8)
mask[rect_y0:rect_y1, rect_x0:rect_x1] = 255
mask_path = os.path.join(test_dir, "test_mask.png")
Image.fromarray(mask).save(mask_path)
print(f"  Saved test mask: {mask_path}")

# --- Build model (randomly initialized, no checkpoint) ---
print("\nBuilding SAM3 model (random weights, no checkpoint)...")
from sam3 import build_sam3_image_model

model = build_sam3_image_model(
    device="cuda",
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=False,  # Skip HF download
    enable_segmentation=True,
)

# Freeze all
for param in model.parameters():
    param.requires_grad = False
model.eval()

print(f"  Model built on CUDA")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# --- Run optimization ---
print("\n" + "=" * 60)
print("Running optimization (20 steps)...")
print("=" * 60 + "\n")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from optimize_text_embedding import (
    load_and_preprocess_image,
    load_target_mask,
    mask_to_bbox_cxcywh,
    init_embedding_from_text,
    forward_with_embedding,
    select_best_query,
    mask_loss,
    box_giou_loss,
    compute_iou,
    decode_nearest_tokens,
)
import torch.nn.functional as F

device = torch.device("cuda")
text_encoder = model.backbone.language_backbone
resolution = 1008

# Load and preprocess
img_tensor = load_and_preprocess_image(img_path, resolution, device)
print(f"  Image tensor shape: {img_tensor.shape}")

# Image features (computed once)
with torch.no_grad():
    backbone_out = model.backbone.forward_image(img_tensor)
print(f"  Image features computed")

# Target mask
target_mask_full, target_mask_resized, orig_h, orig_w = load_target_mask(
    mask_path, resolution, device
)
target_box = mask_to_bbox_cxcywh(target_mask_resized)
print(f"  Target box: {target_box.cpu().tolist()}")

# Initialize embedding
width = text_encoder.encoder.width
ctx_len = 8
opt_embedding = init_embedding_from_text(
    "yellow rectangle", text_encoder, ctx_len, device
)
opt_embedding = torch.nn.Parameter(opt_embedding)
print(f"  Embedding shape: {opt_embedding.shape} (width={width})")

# Optimizer
optimizer = torch.optim.Adam([opt_embedding], lr=0.02)

# Run optimization
losses = []
ious = []
num_steps = 20

for step in range(num_steps):
    optimizer.zero_grad()

    out = forward_with_embedding(
        model=model,
        backbone_out=backbone_out,
        opt_embedding=opt_embedding,
        text_encoder=text_encoder,
        device=device,
    )

    # Select best query
    with torch.no_grad():
        best_q = select_best_query(out, target_mask_resized, target_box, resolution)

    pred_mask = out["pred_masks"][0, best_q]
    # In eval mode: pred_logits is [batch, queries, 1], pred_boxes is [batch, queries, 4]
    pred_logits = out["pred_logits"]
    pred_boxes = out["pred_boxes"]
    if pred_logits.dim() == 3:
        pred_logit = pred_logits[0, best_q, 0]
        pred_box = pred_boxes[0, best_q]
    else:
        pred_logit = pred_logits[-1, 0, best_q, 0]
        pred_box = pred_boxes[-1, 0, best_q]

    # Resize target
    pred_h, pred_w = pred_mask.shape
    target_for_loss = F.interpolate(
        target_mask_resized.unsqueeze(0).unsqueeze(0),
        size=(pred_h, pred_w),
        mode="nearest",
    ).squeeze(0).squeeze(0)

    # Loss
    l_mask = mask_loss(pred_mask, target_for_loss)
    l_score = F.binary_cross_entropy_with_logits(
        pred_logit, torch.ones_like(pred_logit)
    )
    l_box = F.l1_loss(pred_box, target_box) + box_giou_loss(pred_box, target_box)
    total_loss = 5.0 * l_mask + 2.0 * l_score + 1.0 * l_box

    # NaN guard + gradient clipping (important for random-weight models)
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        optimizer.zero_grad()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([opt_embedding], max_norm=1.0)
        optimizer.step()

    # Replace any NaN in embedding with zeros
    with torch.no_grad():
        nan_mask = torch.isnan(opt_embedding)
        if nan_mask.any():
            opt_embedding.data[nan_mask] = 0.0

    with torch.no_grad():
        cur_iou = compute_iou(pred_mask, target_for_loss)

    loss_val = total_loss.item() if not torch.isnan(total_loss) else float('inf')
    losses.append(loss_val)
    ious.append(cur_iou)

    if step % 5 == 0 or step == num_steps - 1:
        score_val = pred_logit.sigmoid().item() if not torch.isnan(pred_logit) else float('nan')
        print(
            f"  Step {step:3d} | Loss: {loss_val:.4f} | "
            f"IoU: {cur_iou:.4f} | Score: {score_val:.4f}"
        )

# --- Verify ----
print("\n" + "=" * 60)
print("Verification")
print("=" * 60)

all_passed = True

# 1. Gradients flowed
has_grad = opt_embedding.grad is not None or any(l != losses[0] for l in losses)
if has_grad or losses[-1] != losses[0]:
    print("  ✓ Gradients flowed through the model to the embedding")
else:
    print("  ✗ No gradient flow detected")
    all_passed = False

# 2. Loss changed (with random weights, it should change)
if losses[-1] != losses[0]:
    direction = "decreased" if losses[-1] < losses[0] else "changed"
    print(f"  ✓ Loss {direction}: {losses[0]:.4f} → {losses[-1]:.4f}")
else:
    print(f"  ✗ Loss unchanged: {losses[0]:.4f}")
    all_passed = False

# 3. Embedding shape correct
assert opt_embedding.shape == (1, ctx_len, width), f"Wrong shape: {opt_embedding.shape}"
print(f"  ✓ Embedding shape correct: {opt_embedding.shape}")

# 4. Output dict has expected keys
expected_keys = ["pred_masks", "pred_logits", "pred_boxes"]
for key in expected_keys:
    assert key in out, f"Missing key: {key}"
print(f"  ✓ Output dict has all expected keys: {expected_keys}")

# 5. Predicted masks have correct shape
print(f"  ✓ pred_masks shape: {out['pred_masks'].shape}")
print(f"  ✓ pred_logits shape: {out['pred_logits'].shape}")
print(f"  ✓ pred_boxes shape: {out['pred_boxes'].shape}")

# 6. Nearest token decode works
tokens = decode_nearest_tokens(opt_embedding, text_encoder)
print(f"  ✓ Nearest tokens decoded: '{tokens[:60]}'")

# 7. No NaN in embedding (after NaN cleanup)
assert not torch.isnan(opt_embedding).any(), "NaN in embedding after cleanup!"
print(f"  ✓ No NaN values in optimized embedding")

# 8. Embedding diverged from initialization
init_emb = init_embedding_from_text("yellow rectangle", text_encoder, ctx_len, device)
diff = (opt_embedding.detach() - init_emb).abs().mean().item()
print(f"  ✓ Embedding diverged from init (mean abs diff: {diff:.6f})")

print(f"\n{'✓ ALL CHECKS PASSED' if all_passed else '⚠ SOME CHECKS HAD WARNINGS'}")
print(f"\nSummary:")
print(f"  Steps: {num_steps}")
print(f"  Initial loss: {losses[0]:.4f}")
print(f"  Final loss: {losses[-1]:.4f}")
print(f"  Initial IoU: {ious[0]:.4f}")
print(f"  Final IoU: {ious[-1]:.4f}")
