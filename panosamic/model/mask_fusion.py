"""
Utilities for fusing masks from dual-view panoramic segmentation.

This module handles:
- Unshifting rotated masks back to original coordinates
- Quality-based mask selection using SAM's predicted IoU and stability scores
- Cross-view mask-based NMS to remove duplicates (handles panoramic seams)
- Cascading tiebreaker logic: quality → area
- Merging mask sets from multiple views

The fusion uses intelligent selection to keep higher-quality masks when duplicates
are detected between the unshifted and shifted (180° rotated) views. This improves
mask quality at panoramic boundaries where objects may be split.

Author: Mahdi Chamseddine
"""

from typing import Any

import torch


def calculate_mask_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    """
    Calculate IoU between two binary masks.

    This is more robust than box IoU for panoramic images where objects
    can be split across the seam (0° / 360° boundary).

    Args:
        mask1: Binary mask, shape (H, W)
        mask2: Binary mask, shape (H, W)

    Returns:
        IoU score in [0, 1]
    """
    intersection = torch.logical_and(mask1, mask2).sum()
    union = torch.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def cross_view_mask_nms(
    masks1: list[dict[str, Any]],
    masks2: list[dict[str, Any]],
    iou_threshold: float = 0.5,
    use_quality_selection: bool = True,
    use_area_tiebreaker: bool = True,
) -> list[dict[str, Any]]:
    """
    Apply mask-based NMS across two sets of masks to remove duplicates.

    Uses direct mask IoU instead of box IoU to handle panoramic seam cases
    where objects are split across the 0°/360° boundary.

    Selection logic for overlapping masks (when use_quality_selection=True):
    1. Compare combined quality scores (from '_quality' field)
    2. If quality difference < epsilon: Use area (prefer larger mask)

    When use_quality_selection=False, uses priority-based selection (masks1 always wins).

    Args:
        masks1: First set of masks (typically from unshifted view)
        masks2: Second set of masks (typically from shifted view)
        iou_threshold: Mask IoU threshold for considering masks as duplicates
        use_quality_selection: Enable quality-based selection instead of priority
        use_area_tiebreaker: Use mask area as tiebreaker

    Returns:
        Merged list of masks with duplicates removed
    """
    if len(masks1) == 0:
        return masks2
    if len(masks2) == 0:
        return masks1

    QUALITY_EPSILON = 0.01  # Threshold for "equal" quality

    keep_mask2 = []
    replace_mask1_with = {}  # Maps mask1 index to mask2 dict for replacement

    for mask2_dict in masks2:
        mask2 = mask2_dict["segmentation"]

        # Find best matching mask from masks1
        max_iou = 0.0
        best_mask1_idx = -1

        for j, mask1_dict in enumerate(masks1):
            mask1 = mask1_dict["segmentation"]
            iou = calculate_mask_iou(mask1, mask2)
            if iou > max_iou:
                max_iou = iou
                best_mask1_idx = j

        # Check if significant overlap exists
        if max_iou > iou_threshold and best_mask1_idx >= 0:
            mask1_dict = masks1[best_mask1_idx]

            if use_quality_selection:
                # Compare quality scores
                quality1 = mask1_dict.get("_quality", 0.0)
                quality2 = mask2_dict.get("_quality", 0.0)
                quality_diff = quality2 - quality1

                should_replace = False

                if abs(quality_diff) > QUALITY_EPSILON:
                    # Clear quality winner
                    should_replace = quality_diff > 0
                elif use_area_tiebreaker:
                    # Quality tie: use area (prefer larger)
                    should_replace = mask2_dict["area"] > mask1_dict["area"]

                if should_replace:
                    replace_mask1_with[best_mask1_idx] = mask2_dict

                keep_mask2.append(False)  # Already handled
            else:
                # Priority mode: always keep mask1
                keep_mask2.append(False)
        else:
            # No overlap, keep mask2
            keep_mask2.append(True)

    # Build final list: Replace masks from masks1, add non-overlapping from masks2
    merged_masks = []
    for i, mask1_dict in enumerate(masks1):
        if i in replace_mask1_with:
            merged_masks.append(replace_mask1_with[i])
        else:
            merged_masks.append(mask1_dict)

    for i, keep in enumerate(keep_mask2):
        if keep:
            merged_masks.append(masks2[i])

    return merged_masks


def fuse_dual_view_masks(
    unshifted_masks: list[dict[str, Any]],
    shifted_masks: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Fuse masks from unshifted and shifted views of a panorama.

    Uses quality-based selection to keep the best mask when duplicates are detected.

    Args:
        unshifted_masks: Masks from original view (dicts with quality metrics)
        shifted_masks: Masks from 180° rotated view (dicts with quality metrics)
        image_width: Width of the panorama image
        iou_threshold: Mask IoU threshold for cross-view NMS

    Returns:
        Merged list of masks from both views with duplicates removed
        Each mask dict contains: {'segmentation': tensor, 'area': int,
                                  'predicted_iou': float, 'stability_score': float}
    """
    # ============ EXPERIMENTAL FLAGS ============
    # These can be modified for rapid experimentation
    USE_QUALITY_SELECTION = True  # Enable quality-based selection
    QUALITY_ALPHA = 1  # Weight for predicted_iou
    QUALITY_BETA = 0  # Weight for stability_score
    USE_AREA_TIEBREAKER = True  # Use mask area as tiebreaker
    # ==========================================

    # Calculate combined quality score for each mask
    def compute_quality_score(mask_dict: dict[str, Any]) -> float:
        if not USE_QUALITY_SELECTION:
            return 0.0  # Not used in priority mode

        pred_iou = mask_dict.get("predicted_iou", 0.0)
        stability = mask_dict.get("stability_score", 0.0)
        return QUALITY_ALPHA * pred_iou + QUALITY_BETA * stability

    # Unshift the rotated masks back to original coordinates
    unshifted_shifted_masks = []

    for mask_dict in shifted_masks:
        mask = mask_dict["segmentation"]
        unshifted_mask = torch.roll(mask, shifts=int(-mask.shape[-1] // 2), dims=-1)

        # Only keep non-empty masks
        if mask_dict["area"] > 0:
            unshifted_shifted_masks.append(
                {
                    "segmentation": unshifted_mask,
                    "area": mask_dict["area"],
                    "predicted_iou": mask_dict.get("predicted_iou", 0.0),
                    "stability_score": mask_dict.get("stability_score", 0.0),
                }
            )

    # Pre-compute quality scores for all masks
    for mask_dict in unshifted_masks:
        mask_dict["_quality"] = compute_quality_score(mask_dict)

    for mask_dict in unshifted_shifted_masks:
        mask_dict["_quality"] = compute_quality_score(mask_dict)

    # Apply cross-view mask-based NMS to remove duplicates
    # This handles panoramic seam cases where objects are split
    merged_masks = cross_view_mask_nms(
        unshifted_masks,
        unshifted_shifted_masks,
        iou_threshold,
        use_quality_selection=USE_QUALITY_SELECTION,
        use_area_tiebreaker=USE_AREA_TIEBREAKER,
    )

    # Clean up temporary fields
    for mask_dict in merged_masks:
        mask_dict.pop("_quality", None)

    # Sort by area (descending) for consistent processing order in downstream fusion
    # Larger instances processed first = better quality in semantic refinement
    merged_masks.sort(key=lambda x: x["area"], reverse=True)
    return merged_masks
