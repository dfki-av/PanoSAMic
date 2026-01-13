"""
Post-processing utilities for instance segmentation masks.

This module implements the post-processing steps used by SAM's AutomaticMaskGenerator:
- Stability score calculation (from SAM)
- IoU-based filtering
- Box-based Non-Maximum Suppression (uses torchvision's optimized batched_nms)
- Small region removal (custom PyTorch implementation without OpenCV dependency)

These steps filter raw mask predictions to produce high-quality, non-redundant instance masks.

Author: Mahdi Chamseddine
"""

from typing import Any

import torch
from segment_anything.utils.amg import (
    batched_mask_to_box,
    calculate_stability_score,
)
from torchvision.ops.boxes import batched_nms


def connected_components_torch(mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    Label connected components in a binary mask using iterative 4-connectivity.

    Args:
        mask: Binary mask, shape (H, W)

    Returns:
        Tuple of (labeled_mask, num_components)
        - labeled_mask: Labeled mask with component IDs, shape (H, W)
        - num_components: Number of connected components
    """
    H, W = mask.shape
    labeled = torch.zeros_like(mask, dtype=torch.int32)
    current_label = 0

    # Find all True pixels
    true_pixels = mask.nonzero(as_tuple=False)

    if len(true_pixels) == 0:
        return labeled, 0

    # Convert to set for fast lookup
    unvisited = set(map(tuple, true_pixels.tolist()))

    while unvisited:
        # Start new component
        current_label += 1
        start_pixel = next(iter(unvisited))
        unvisited.remove(start_pixel)

        # BFS to find all connected pixels
        queue = [start_pixel]
        component_pixels = []

        while queue:
            pixel = queue.pop(0)
            y, x = pixel
            component_pixels.append((y, x))

            # Check 4 neighbors
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                neighbor = (ny, nx)

                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    queue.append(neighbor)

        # Label all pixels in this component
        for y, x in component_pixels:
            labeled[y, x] = current_label

    return labeled, current_label


def remove_small_regions(
    masks: torch.Tensor,
    min_area: int,
) -> tuple[torch.Tensor, list[bool]]:
    """
    Remove small disconnected regions and holes from masks.

    Args:
        masks: Binary masks (torch.Tensor), shape (N, H, W)
        min_area: Minimum area in pixels for a region to be kept

    Returns:
        Tuple of (cleaned_masks, changed_flags)
        - cleaned_masks: Masks with small regions removed, shape (N, H, W)
        - changed_flags: Boolean flags indicating which masks were modified
    """
    cleaned_masks = []
    changed_flags = []

    for mask in masks:
        # Label connected components
        labeled, num_features = connected_components_torch(mask)

        if num_features == 0:
            cleaned_masks.append(mask)
            changed_flags.append(False)
            continue

        # Calculate area of each component
        areas = []
        for i in range(1, num_features + 1):
            area = (labeled == i).sum().item()
            areas.append(area)

        # Keep only large components
        large_components = [i + 1 for i, area in enumerate(areas) if area >= min_area]

        if len(large_components) == 0:
            # All regions too small - return empty mask
            cleaned_mask = torch.zeros_like(mask, dtype=torch.bool)
            changed_flags.append(True)
        elif len(large_components) == num_features:
            # All regions large enough - no change
            cleaned_mask = mask
            changed_flags.append(False)
        else:
            # Some regions removed
            cleaned_mask = torch.zeros_like(labeled, dtype=torch.bool)
            for comp_id in large_components:
                cleaned_mask |= labeled == comp_id
            changed_flags.append(True)

        cleaned_masks.append(cleaned_mask)

    return torch.stack(cleaned_masks), changed_flags


def postprocess_instances(
    masks: torch.Tensor,
    iou_predictions: torch.Tensor,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    stability_score_offset: float = 1.0,
    box_nms_thresh: float = 0.7,
    min_mask_region_area: int = 0,
    mask_threshold: float = 0.0,
    batch_size: int | None = None,
) -> list[dict[str, Any]]:
    """
    Apply full post-processing pipeline to raw mask predictions.

    This matches the processing done by SAM's AutomaticMaskGenerator, with support
    for batched processing of expensive operations (stability score calculation and
    small region removal) while maintaining global NMS.

    Args:
        masks: Raw mask predictions, shape (N, H, W), values in [0, 1]
        iou_predictions: Predicted IoU scores, shape (N,)
        pred_iou_thresh: Filter masks with predicted IoU below this threshold
        stability_score_thresh: Filter masks with stability score below this threshold
        stability_score_offset: Offset for stability score calculation
        box_nms_thresh: IoU threshold for box NMS
        min_mask_region_area: Minimum area for connected components
        mask_threshold: Threshold for binarizing masks
        batch_size: If provided, process expensive operations in batches to save memory.
                   NMS is always applied globally. Default None processes all at once.

    Returns:
        List of dictionaries, each containing:
        - segmentation: Binary mask (H, W) as torch.Tensor
        - area: Mask area in pixels (int)
        - predicted_iou: Model's predicted IoU score (float in [0, 1])
        - stability_score: Mask stability score (float in [0, 1])

    Note:
        Quality metrics (predicted_iou, stability_score) are preserved
        for use in quality-based mask fusion during dual-view panoramic segmentation.
    """

    # 1. Filter by predicted IoU
    iou_mask = iou_predictions >= pred_iou_thresh
    if not iou_mask.any():
        return []

    masks_filtered = masks[iou_mask]
    iou_predictions_filtered = iou_predictions[iou_mask]

    # 2. Calculate stability scores (can be batched for memory efficiency)
    if batch_size is not None and len(masks_filtered) > batch_size:
        stability_scores_list = []
        for i in range(0, len(masks_filtered), batch_size):
            batch = masks_filtered[i : i + batch_size]
            scores = calculate_stability_score(
                masks=batch,
                mask_threshold=mask_threshold,
                threshold_offset=stability_score_offset,
            )
            stability_scores_list.append(scores)
        stability_scores = torch.cat(stability_scores_list, dim=0)
    else:
        stability_scores = calculate_stability_score(
            masks=masks_filtered,
            mask_threshold=mask_threshold,
            threshold_offset=stability_score_offset,
        )

    # 3. Filter by stability score
    stability_mask = stability_scores >= stability_score_thresh
    if not stability_mask.any():
        return []

    masks_stable = masks_filtered[stability_mask]
    iou_predictions_stable = iou_predictions_filtered[stability_mask]
    stability_scores_stable = stability_scores[stability_mask]

    # 4. Binarize and calculate bounding boxes (batched to avoid OOM on thresholding)
    if batch_size is not None and len(masks_stable) > batch_size:
        boxes_list = []
        # Process in batches to compute boxes without storing all binary masks
        for i in range(0, len(masks_stable), batch_size):
            batch = masks_stable[i : i + batch_size]
            batch_binary = batch > mask_threshold
            batch_boxes = batched_mask_to_box(batch_binary)
            boxes_list.append(batch_boxes)
            del batch_binary  # Free immediately
        boxes = torch.cat(boxes_list, dim=0)
        del boxes_list
    else:
        # Non-batched: create full binary tensor (small enough to fit in memory)
        masks_binary = masks_stable > mask_threshold
        boxes = batched_mask_to_box(masks_binary)

    # 5. Apply box NMS globally (ALWAYS on full set - never batched)
    keep_indices = (
        torch.zeros(0, dtype=torch.long, device=boxes.device)
        if len(iou_predictions_stable) == 0
        else batched_nms(
            boxes=boxes.float(),
            scores=iou_predictions_stable,
            idxs=torch.zeros_like(iou_predictions_stable, dtype=torch.long),
            iou_threshold=box_nms_thresh,
        )
    )

    if len(keep_indices) == 0:
        return []

    # Get binary masks for kept indices
    if batch_size is not None and len(masks_stable) > batch_size:
        # Deferred binarization: only binarize the masks we're keeping (much smaller set)
        masks_nms = masks_stable[keep_indices] > mask_threshold
        del masks_stable
    else:
        # Use pre-computed binary masks
        masks_nms = masks_binary[keep_indices]  # type: ignore
        del masks_binary, masks_stable  # type: ignore

    iou_predictions_nms = iou_predictions_stable[keep_indices]
    stability_scores_nms = stability_scores_stable[keep_indices]

    # 6. Remove small regions (can be batched for memory efficiency)
    if min_mask_region_area > 0:
        if batch_size is not None and len(masks_nms) > batch_size:
            masks_cleaned_list = []
            for i in range(0, len(masks_nms), batch_size):
                batch = masks_nms[i : i + batch_size]
                cleaned, _ = remove_small_regions(batch, min_mask_region_area)
                masks_cleaned_list.append(cleaned)
            masks_cleaned = torch.cat(masks_cleaned_list, dim=0)
        else:
            masks_cleaned, _ = remove_small_regions(masks_nms, min_mask_region_area)
    else:
        masks_cleaned = masks_nms

    # 7. Remove empty masks
    non_empty_mask = masks_cleaned.sum(dim=(-1, -2)) > 0
    if not non_empty_mask.any():
        return []

    masks_final = masks_cleaned[non_empty_mask]
    iou_predictions_final = iou_predictions_nms[non_empty_mask]
    stability_scores_final = stability_scores_nms[non_empty_mask]

    # 8. Build final results
    results = []
    for i, mask in enumerate(masks_final):
        results.append(
            {
                "segmentation": mask,
                "area": int(mask.sum()),
                "predicted_iou": float(iou_predictions_final[i]),
                "stability_score": float(stability_scores_final[i]),
            }
        )

    return results


def merge_masks_box_nms(
    masks: list[dict[str, Any]], iou_threshold: float
) -> list[dict[str, Any]]:
    """
    Merge masks from multiple modalities/views using box-based NMS.

    Uses torchvision's batched_nms for efficient parallel processing.
    This is faster than mask-based NMS but less accurate for panoramic seams.

    Args:
        masks: List of mask dictionaries to merge
        iou_threshold: Box IoU threshold for NMS

    Returns:
        Filtered list of masks with duplicates removed, sorted by area
    """
    if len(masks) == 0:
        return []

    # Compute bounding boxes from segmentation masks
    boxes_list = []
    for m in masks:
        mask = m["segmentation"]
        # Find non-zero coordinates
        ys, xs = torch.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            # Empty mask - use a small dummy box
            boxes_list.append([0, 0, 1, 1])
        else:
            x1, x2 = xs.min().item(), xs.max().item()
            y1, y2 = ys.min().item(), ys.max().item()
            boxes_list.append([x1, y1, x2, y2])

    boxes_xyxy = torch.tensor(
        boxes_list, dtype=torch.float32
    )  # [N, 4] in x1,y1,x2,y2 format

    # Compute quality scores (predicted_iou * stability_score)
    scores = torch.tensor(
        [m.get("predicted_iou", 0.0) * m.get("stability_score", 0.0) for m in masks],
        dtype=torch.float32,
    )

    # Use batched_nms with all masks in same class (idxs=0)
    # This applies NMS across all masks regardless of modality
    idxs = torch.zeros(len(masks), dtype=torch.int64)  # All same class
    keep_indices = batched_nms(boxes_xyxy, scores, idxs, iou_threshold)

    # Keep only selected masks
    kept_masks = [masks[i] for i in keep_indices.tolist()]

    # Sort by area for consistent downstream processing
    kept_masks.sort(key=lambda x: x["area"], reverse=True)
    return kept_masks


def merge_masks_greedy(
    masks: list[dict[str, Any]], iou_threshold: float
) -> list[dict[str, Any]]:
    """
    Merge masks from multiple modalities/views using mask-based greedy NMS.

    Uses direct mask IoU computation for more accurate duplicate detection.
    This is slower than box-based NMS but more accurate for panoramic seams
    where objects can be split.

    Args:
        masks: List of mask dictionaries to merge
        iou_threshold: Mask IoU threshold for NMS

    Returns:
        Filtered list of masks with duplicates removed, sorted by area
    """
    if len(masks) == 0:
        return []

    # Import here to avoid circular dependency
    from panosamic.model.mask_fusion import calculate_mask_iou

    # Sort by quality (predicted_iou * stability_score) descending
    masks_sorted = sorted(
        masks,
        key=lambda m: m.get("predicted_iou", 0.0) * m.get("stability_score", 0.0),
        reverse=True,
    )

    # Greedy NMS: keep high-quality masks, remove overlapping lower-quality ones
    kept_masks = []
    for candidate in masks_sorted:
        candidate_mask = candidate["segmentation"]

        # Check if it overlaps significantly with any kept mask
        overlaps = False
        for kept in kept_masks:
            kept_mask = kept["segmentation"]
            iou = calculate_mask_iou(candidate_mask, kept_mask)
            if iou > iou_threshold:
                overlaps = True
                break

        if not overlaps:
            kept_masks.append(candidate)

    # Sort by area for consistent downstream processing
    kept_masks.sort(key=lambda x: x["area"], reverse=True)
    return kept_masks
