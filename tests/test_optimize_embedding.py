# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Unit tests for the optimize_text_embedding script (CPU-only, no model download)."""

import sys
import os

import numpy as np
import torch
import torch.nn.functional as F

# Add scripts to path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from optimize_text_embedding import (
    box_giou_loss,
    compute_iou,
    dice_loss,
    mask_loss,
    mask_to_bbox_cxcywh,
)


class TestLossFunctions:
    """Test the individual loss functions used in optimization."""

    def test_dice_loss_perfect_match(self):
        """Dice loss should be ~0 for a perfect match."""
        # Large positive logits → sigmoid ≈ 1
        pred = torch.ones(10, 10) * 10.0
        target = torch.ones(10, 10)
        loss = dice_loss(pred, target)
        assert loss.item() < 0.05, f"Dice loss for perfect match should be ~0, got {loss.item()}"

    def test_dice_loss_no_overlap(self):
        """Dice loss should be ~1 for no overlap."""
        pred = torch.ones(10, 10) * 10.0  # all foreground
        target = torch.zeros(10, 10)       # all background
        loss = dice_loss(pred, target)
        assert loss.item() > 0.9, f"Dice loss for no overlap should be ~1, got {loss.item()}"

    def test_mask_loss_differentiable(self):
        """Mask loss should be differentiable."""
        pred = torch.randn(10, 10, requires_grad=True)
        target = torch.ones(10, 10)
        loss = mask_loss(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    def test_mask_loss_decreases_toward_target(self):
        """Loss should be lower for predictions closer to the target."""
        target = torch.ones(10, 10)
        # Close prediction (high logits where target=1)
        pred_close = torch.ones(10, 10) * 5.0
        # Far prediction (negative logits where target=1)
        pred_far = torch.ones(10, 10) * -5.0
        loss_close = mask_loss(pred_close, target)
        loss_far = mask_loss(pred_far, target)
        assert loss_close < loss_far

    def test_box_giou_loss_perfect_match(self):
        """GIoU loss should be 0 for identical boxes."""
        box = torch.tensor([0.5, 0.5, 0.3, 0.3])
        loss = box_giou_loss(box, box)
        assert abs(loss.item()) < 0.01, f"GIoU loss for matching boxes should be ~0, got {loss.item()}"

    def test_box_giou_loss_no_overlap(self):
        """GIoU loss should be > 1 for non-overlapping boxes."""
        box1 = torch.tensor([0.1, 0.1, 0.1, 0.1])
        box2 = torch.tensor([0.9, 0.9, 0.1, 0.1])
        loss = box_giou_loss(box1, box2)
        assert loss.item() > 1.0

    def test_box_giou_loss_differentiable(self):
        """GIoU loss should be differentiable."""
        pred = torch.tensor([0.5, 0.5, 0.3, 0.3], requires_grad=True)
        target = torch.tensor([0.6, 0.6, 0.3, 0.3])
        loss = box_giou_loss(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestIoU:
    """Test IoU computation."""

    def test_iou_perfect_match(self):
        pred = torch.ones(10, 10) * 10.0  # high logits
        target = torch.ones(10, 10)
        iou = compute_iou(pred, target)
        assert iou > 0.99

    def test_iou_no_overlap(self):
        pred = torch.ones(10, 10) * -10.0  # negative logits → all background
        target = torch.ones(10, 10)
        iou = compute_iou(pred, target)
        assert iou < 0.01

    def test_iou_partial_overlap(self):
        pred = torch.zeros(10, 10) * -10.0
        pred[:5, :] = 10.0  # top half positive
        target = torch.zeros(10, 10)
        target[:, :5] = 1.0  # left half target
        iou = compute_iou(pred, target)
        assert 0.1 < iou < 0.5


class TestTargetMask:
    """Test target mask utilities."""

    def test_mask_to_bbox_cxcywh_centered(self):
        """A centered square mask should produce a centered bbox."""
        mask = torch.zeros(100, 100)
        mask[25:75, 25:75] = 1.0
        bbox = mask_to_bbox_cxcywh(mask)
        assert abs(bbox[0].item() - 0.5) < 0.05  # cx ≈ 0.5
        assert abs(bbox[1].item() - 0.5) < 0.05  # cy ≈ 0.5
        assert abs(bbox[2].item() - 0.5) < 0.05  # w ≈ 0.5
        assert abs(bbox[3].item() - 0.5) < 0.05  # h ≈ 0.5

    def test_mask_to_bbox_empty_mask(self):
        """Empty mask should return a default full-image bbox."""
        mask = torch.zeros(100, 100)
        bbox = mask_to_bbox_cxcywh(mask)
        assert bbox.shape == (4,)

    def test_mask_to_bbox_corner(self):
        """A mask in the top-left corner should produce a top-left bbox."""
        mask = torch.zeros(100, 100)
        mask[:20, :20] = 1.0
        bbox = mask_to_bbox_cxcywh(mask)
        assert bbox[0].item() < 0.2  # cx in left region
        assert bbox[1].item() < 0.2  # cy in top region


class TestEmbeddingGradient:
    """Test that gradients flow through the optimization path."""

    def test_embedding_gradient_through_simple_transform(self):
        """Verify a learnable embedding can receive gradients through a linear layer."""
        embedding = torch.randn(1, 16, 1024, requires_grad=True)
        linear = torch.nn.Linear(1024, 256)
        target = torch.randn(1, 16, 256)

        output = linear(embedding)
        loss = F.mse_loss(output, target)
        loss.backward()

        assert embedding.grad is not None
        assert embedding.grad.shape == (1, 16, 1024)
        assert not torch.isnan(embedding.grad).any()

    def test_one_step_reduces_loss(self):
        """One optimization step on a simple path should reduce loss."""
        embedding = torch.randn(1, 16, 1024, requires_grad=True)
        linear = torch.nn.Linear(1024, 256)
        linear.eval()
        for p in linear.parameters():
            p.requires_grad = False

        target = torch.randn(1, 16, 256)
        optimizer = torch.optim.Adam([embedding], lr=0.1)

        # Step 1
        output1 = linear(embedding)
        loss1 = F.mse_loss(output1, target)
        loss1.backward()
        optimizer.step()

        # Step 2
        optimizer.zero_grad()
        output2 = linear(embedding)
        loss2 = F.mse_loss(output2, target)

        assert loss2.item() < loss1.item(), (
            f"Loss should decrease: {loss1.item()} -> {loss2.item()}"
        )
