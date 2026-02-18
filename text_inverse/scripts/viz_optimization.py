# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
Visualization utility for token embedding optimization results.

Loads a completed optimization run and generates detailed visualizations:
- Loss curves
- Mask progression over optimization steps
- Nearest token decode at each step

Usage:
    python scripts/viz_optimization.py \
        --run-dir outputs/optimization_run \
        --output viz_output.png
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize token embedding optimization results"
    )
    parser.add_argument(
        "--run-dir", type=str, required=True, help="Optimization run output directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: <run-dir>/summary.png)",
    )
    return parser.parse_args()


def visualize_run(args):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    run_dir = Path(args.run_dir)

    # Load history
    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        print(f"Error: {hist_path} not found")
        return
    with open(hist_path) as f:
        history = json.load(f)

    # Load embedding info
    emb_path = run_dir / "optimized_embedding.pt"
    emb_info = {}
    if emb_path.exists():
        emb_data = torch.load(emb_path, map_location="cpu", weights_only=True)
        emb_info = {
            "best_iou": emb_data.get("best_iou", 0),
            "seed_text": emb_data.get("seed_text", "N/A"),
            "num_steps": emb_data.get("num_steps", 0),
            "context_length": emb_data.get("context_length", 0),
        }

    # Load nearest tokens
    tokens_path = run_dir / "nearest_tokens.txt"
    nearest_tokens = ""
    if tokens_path.exists():
        nearest_tokens = tokens_path.read_text().strip()

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # ---- Loss curves ----
    ax_loss = fig.add_subplot(gs[0, 0:2])
    ax_loss.plot(history["step"], history["loss"], label="Total", color="red", linewidth=2)
    ax_loss.plot(history["step"], history["mask_loss"], label="Mask", alpha=0.7)
    ax_loss.plot(history["step"], history["score_loss"], label="Score", alpha=0.7)
    ax_loss.plot(history["step"], history["box_loss"], label="Box", alpha=0.7)
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss Curves")
    ax_loss.legend(loc="upper right")
    ax_loss.grid(True, alpha=0.3)

    # ---- IoU curve ----
    ax_iou = fig.add_subplot(gs[0, 2])
    ax_iou.plot(history["step"], history["iou"], color="green", linewidth=2)
    ax_iou.fill_between(history["step"], history["iou"], alpha=0.2, color="green")
    ax_iou.set_xlabel("Step")
    ax_iou.set_ylabel("IoU")
    ax_iou.set_title("Mask IoU")
    ax_iou.grid(True, alpha=0.3)
    ax_iou.set_ylim(0, 1)

    # ---- Final comparison (if available) ----
    final_img_path = run_dir / "final_comparison.png"
    if final_img_path.exists():
        ax_final = fig.add_subplot(gs[1, 0:2])
        final_img = plt.imread(str(final_img_path))
        ax_final.imshow(final_img)
        ax_final.set_title("Final Mask Comparison")
        ax_final.axis("off")

    # ---- Info panel ----
    ax_info = fig.add_subplot(gs[1, 2])
    ax_info.axis("off")
    info_text = (
        f"Optimization Summary\n"
        f"{'─' * 30}\n\n"
        f"Best IoU: {emb_info.get('best_iou', 'N/A'):.4f}\n"
        f"Seed Text: {emb_info.get('seed_text', 'N/A')}\n"
        f"Steps: {emb_info.get('num_steps', 'N/A')}\n"
        f"Context Length: {emb_info.get('context_length', 'N/A')}\n"
        f"Final Loss: {history['loss'][-1]:.4f}\n\n"
        f"Nearest Tokens:\n{nearest_tokens[:100]}"
    )
    ax_info.text(
        0.05, 0.95, info_text,
        transform=ax_info.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Save
    output_path = args.output or str(run_dir / "summary.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to: {output_path}")
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    visualize_run(args)
