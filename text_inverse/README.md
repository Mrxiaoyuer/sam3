# Textual Inversion for SAM3

Learning continuous token embeddings that activate specific segmentation masks in SAM3 via gradient ascent — effectively performing **textual inversion** adapted to segmentation.

## Overview

SAM3 uses a text-conditioned architecture:

```
Text → Tokenize → Embed [1, seq, 1024] → TextTransformer → Resizer → Fusion → Mask Decoder → Masks
```

We optimize directly at the **pre-transformer embedding level** (`[1, seq_len, 1024]`). Given a target mask, we find the embedding that makes the frozen SAM3 model produce that mask, by gradient-ascending through the full model.

This enables:
- **Per-image prompt discovery**: "What text would produce this mask?"
- **Class-level prompt learning**: "What single prompt segments all instances of this class?"
- **Zero-shot transfer**: learned class embeddings generalize to unseen images

## Key Findings

### 1. Per-Image Optimization (CAMO Dataset)

Optimized individual embeddings for 5 camouflaged object images with 3 initialization strategies × 150 steps each.

| Init Strategy | Avg Peak IoU | Best Single | Trend |
|---------------|-------------|-------------|-------|
| "camouflaged animal" | 0.72 | 0.93 (camo_3) | ↑ all |
| "red sports car" (wrong) | 0.69 | 0.90 (camo_1) | ↑ all |
| Random noise | 0.72 | 0.91 (camo_3) | ↑ 14/15 |

**Finding**: All 15/15 runs converge. Random init often outperforms text seeds — the optimizer finds better solutions in unconstrained embedding space.

### 2. Class-Level Embedding (CAMO, 5 images)

Learned a single shared embedding across 5 camouflaged images simultaneously.

| Metric | Start → End |
|--------|-------------|
| Mean IoU | 0.581 → **0.682** |
| Loss | 6.08 → **2.26** |
| Min IoU | 0.381 → **0.463** |
| Max IoU | 0.708 → **0.812** |

**Finding**: One embedding generalizes across diverse scenes. Loss decreases monotonically.

### 3. Multi-Class Case Study (CAMO + Kvasir-SEG)

Trained class embeddings on 50 images/class, evaluated on 20 held-out images/class.

| Class | Train IoU | Val IoU (unseen) | Gen. Gap |
|-------|-----------|-----------------|----------|
| Camouflaged objects (CAMO) | 0.583 ± 0.164 | **0.554 ± 0.162** | +0.029 |
| GI Polyps (Kvasir-SEG) | 0.826 ± 0.070 | **0.841 ± 0.067** | −0.015 |

**Cross-class confusion:**

| Embedding ↓ \ Eval → | CAMO | Polyp |
|-----------------------|------|-------|
| CAMO embedding | **0.554** | 0.808 |
| Polyp embedding | 0.535 | **0.841** |

**Findings**:
- Both classes converge and generalize to unseen images
- Polyps are easier (consistent visual appearance) → 0.84 IoU
- Camouflaged objects are harder (diverse shapes/textures) → 0.55 IoU
- Cross-class confusion is high, suggesting SAM3's text conditioning is partially class-agnostic — embeddings primarily activate "find the foreground object" rather than class-specific features

## Architecture

### Injection Point

We bypass `VETextEncoder.forward()` and inject the learnable embedding directly:

```python
# Standard path (bypassed):
text → tokenizer → token_embedding → TextTransformer → resizer → output

# Our path:
learnable_embedding [1, seq_len, 1024]
    → TextTransformer (frozen)
    → resizer (frozen)
    → output
```

### Loss Function

Composite loss with three components:

```
L = λ_mask · (BCE + Dice) + λ_score · BCE_score + λ_box · (L1 + GIoU)
```

- **Mask loss** (BCE + Dice): pixel-level mask quality
- **Score loss** (BCE): detection confidence
- **Box loss** (L1 + GIoU): bounding box accuracy
- Default weights: `λ_mask=5.0, λ_score=2.0, λ_box=1.0`

### Query Selection

SAM3 produces 200 object queries per image. We select the best query per step via IoU with the target mask as ground truth.

## Repository Structure

```
text_inverse/
├── README.md                           ← this file
├── scripts/
│   ├── optimize_text_embedding.py      ← per-image embedding optimization
│   ├── optimize_class_embedding.py     ← multi-image class embedding
│   ├── case_study_multiclass.py        ← multi-class case study (CAMO + Kvasir)
│   └── viz_optimization.py             ← visualization utility
├── tests/
│   ├── test_optimize_embedding.py      ← 15 CPU-only unit tests
│   └── test_e2e_optimize.py            ← end-to-end GPU test
└── outputs/                            ← experiment artifacts (not in git)
    ├── camo_convergence/               ← per-image CAMO results
    ├── camo_class_result/              ← class-level CAMO results
    ├── case_study_results/             ← multi-class case study
    └── case_study_data/                ← downloaded datasets
```

## Usage

### Per-Image Optimization

```bash
python text_inverse/scripts/optimize_text_embedding.py \
    --image path/to/image.jpg \
    --target-mask path/to/binary_mask.png \
    --seed-text "building" \
    --num-steps 200 \
    --output-dir text_inverse/outputs/my_run
```

### Class-Level Optimization

```bash
python text_inverse/scripts/optimize_class_embedding.py \
    --image-dir path/to/class_images/ \
    --mask-dir path/to/class_masks/ \
    --seed-text "camouflaged animal" \
    --num-steps 300 \
    --batch-size 8 \
    --output-dir text_inverse/outputs/class_run
```

### Multi-Class Case Study

```bash
python text_inverse/scripts/case_study_multiclass.py \
    --max-train 50 \
    --max-val 20 \
    --num-steps 300 \
    --batch-size 8
```

### Tests

```bash
# Unit tests (CPU, no model needed)
python -m pytest text_inverse/tests/test_optimize_embedding.py -v

# End-to-end GPU test
python text_inverse/tests/test_e2e_optimize.py
```

## Outputs

Each optimization run produces:
- `optimized_embedding.pt` / `class_embedding.pt` — the learned embedding tensor
- `history.json` — per-step loss, IoU, and score metrics
- `nearest_tokens.txt` — decoded nearest vocabulary tokens
- `final_comparison.png` — side-by-side mask comparison (per-image only)

## Requirements

- SAM3 model (requires HuggingFace access to `facebook/sam3`)
- CUDA GPU (tested on single GPU, ~5GB VRAM)
- Python packages: `torch`, `numpy`, `PIL`, `torchvision`, `matplotlib`
- For case study: `datasets` (HuggingFace)

```bash
export HF_TOKEN=your_token_here
uv pip install datasets  # for case study only
```
