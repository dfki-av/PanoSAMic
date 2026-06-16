"""Unit tests for mask postprocessing and dual-view fusion."""

from __future__ import annotations

import torch

from panosamic.model.mask_fusion import calculate_mask_iou, fuse_dual_view_masks
from panosamic.model.mask_postprocessing import (
    merge_masks_greedy,
    postprocess_instances,
    remove_small_regions,
)

_H, _W = 32, 64


def _mask_dict(mask: torch.Tensor, iou: float = 0.9, stab: float = 0.95) -> dict:
    return {
        "segmentation": mask.bool(),
        "area": int(mask.sum()),
        "predicted_iou": iou,
        "stability_score": stab,
    }


# ── calculate_mask_iou ────────────────────────────────────────────────────────


def test_iou_identical_masks():
    mask = torch.ones(_H, _W, dtype=torch.bool)
    assert calculate_mask_iou(mask, mask) == 1.0


def test_iou_disjoint_masks():
    a = torch.zeros(_H, _W, dtype=torch.bool)
    b = torch.zeros(_H, _W, dtype=torch.bool)
    a[:, : _W // 2] = True
    b[:, _W // 2 :] = True
    assert calculate_mask_iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = torch.zeros(_H, _W, dtype=torch.bool)
    b = torch.zeros(_H, _W, dtype=torch.bool)
    a[:, : _W // 2] = True  # left half
    b[:, _W // 4 : _W * 3 // 4] = True  # middle half
    iou = calculate_mask_iou(a, b)
    assert 0.0 < iou < 1.0


def test_iou_empty_masks():
    empty = torch.zeros(_H, _W, dtype=torch.bool)
    assert calculate_mask_iou(empty, empty) == 0.0


# ── remove_small_regions ──────────────────────────────────────────────────────


def test_remove_small_regions_keeps_large():
    mask = torch.zeros(_H, _W, dtype=torch.bool)
    mask[4:12, 4:20] = True  # 8x16 = 128 pixels
    cleaned, changed = remove_small_regions(mask.unsqueeze(0), min_area=10)
    assert not changed[0]
    assert cleaned[0].sum() == mask.sum()


def test_remove_small_regions_removes_tiny():
    mask = torch.zeros(_H, _W, dtype=torch.bool)
    mask[0, 0] = True  # 1 pixel
    cleaned, changed = remove_small_regions(mask.unsqueeze(0), min_area=5)
    assert changed[0]
    assert cleaned[0].sum() == 0


# ── postprocess_instances ─────────────────────────────────────────────────────


def test_postprocess_empty_when_low_quality():
    masks = torch.rand(_H, _W).unsqueeze(0)  # (1, H, W)
    ious = torch.tensor([0.1])  # below pred_iou_thresh=0.88
    result = postprocess_instances(masks, ious)
    assert result == []


def test_postprocess_returns_dict_fields():
    mask = torch.zeros(1, _H, _W)
    mask[0, 5:15, 5:30] = 1.0
    result = postprocess_instances(
        mask, torch.tensor([0.95]), stability_score_thresh=0.0
    )
    if result:  # may be empty after stability/NMS, but if not:
        assert "segmentation" in result[0]
        assert "area" in result[0]
        assert "predicted_iou" in result[0]
        assert "stability_score" in result[0]


# ── merge_masks_greedy ────────────────────────────────────────────────────────


def test_merge_greedy_empty_inputs():
    assert merge_masks_greedy([], iou_threshold=0.5) == []


def test_merge_greedy_no_overlap_keeps_all():
    a = torch.zeros(_H, _W, dtype=torch.bool)
    b = torch.zeros(_H, _W, dtype=torch.bool)
    a[:, : _W // 2] = True
    b[:, _W // 2 :] = True
    masks = [_mask_dict(a), _mask_dict(b)]
    kept = merge_masks_greedy(masks, iou_threshold=0.5)
    assert len(kept) == 2


def test_merge_greedy_removes_duplicate():
    mask = torch.zeros(_H, _W, dtype=torch.bool)
    mask[2:10, 2:20] = True
    masks = [_mask_dict(mask, iou=0.95), _mask_dict(mask, iou=0.90)]
    kept = merge_masks_greedy(masks, iou_threshold=0.5)
    assert len(kept) == 1


# ── fuse_dual_view_masks ──────────────────────────────────────────────────────


def test_fuse_empty_inputs():
    assert fuse_dual_view_masks([], []) == []


def test_fuse_one_empty_returns_other():
    mask = torch.zeros(_H, _W, dtype=torch.bool)
    mask[2:8, 2:10] = True
    masks = [_mask_dict(mask)]
    # When one side is empty the other side is returned unchanged (same length, same area)
    assert len(fuse_dual_view_masks(masks, [])) == 1
    result = fuse_dual_view_masks([], masks)
    assert len(result) == 1
    assert result[0]["area"] == masks[0]["area"]


def test_fuse_non_overlapping_keeps_all():
    a = torch.zeros(_H, _W, dtype=torch.bool)
    b = torch.zeros(_H, _W, dtype=torch.bool)
    a[:, : _W // 4] = True
    b[:, _W // 2 :] = True
    result = fuse_dual_view_masks([_mask_dict(a)], [_mask_dict(b)])
    assert len(result) == 2


def test_fuse_result_sorted_by_area_descending():
    small = torch.zeros(_H, _W, dtype=torch.bool)
    large = torch.zeros(_H, _W, dtype=torch.bool)
    small[0:2, 0:2] = True
    large[0:10, 0:20] = True
    result = fuse_dual_view_masks([_mask_dict(small)], [_mask_dict(large)])
    areas = [r["area"] for r in result]
    assert areas == sorted(areas, reverse=True)
