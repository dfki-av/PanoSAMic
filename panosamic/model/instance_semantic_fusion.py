"""
Instance-guided semantic segmentation refinement.

This module refines semantic segmentation predictions using high-quality SAM instance masks.
The approach:
1. For each instance mask: assign the majority semantic class within that mask
2. For background pixels (no instance): use the original semantic prediction

This leverages SAM's superior boundary quality to improve semantic segmentation.

Author: Mahdi Chamseddine

Usage (post-processing after model inference):
    outputs = model(batched_input, batched_prompts=[None])
    for output in outputs:
        semantic = output['sem_preds']  # (C, H, W) logits
        instances = output.get('instance_masks', [])

        if len(instances) > 0:
            refined = refine_semantic_with_instances(semantic, instances)
            # Convert to class labels for evaluation
            refined_labels = refined.argmax(dim=0)
"""

from typing import Any

import numpy as np
import torch


def refine_semantic_with_instances(
    semantic_pred: torch.Tensor,
    instance_masks: list[dict[str, Any]],
) -> torch.Tensor:
    """
    Refine semantic segmentation using instance masks (hard assignment).

    For each instance mask, all pixels are assigned the majority semantic class
    within that instance. Background pixels retain their original predictions.

    Args:
        semantic_pred: Semantic predictions, shape (num_classes, H, W) (logits or probabilities)
        instance_masks: List of instance mask dictionaries, each with 'segmentation' key

    Returns:
        Refined semantic predictions, shape (num_classes, H, W) in one-hot format
    """
    if len(instance_masks) == 0:
        # No instances, return original predictions
        return semantic_pred

    device = semantic_pred.device
    C, H, W = semantic_pred.shape

    # Get class predictions from semantic logits/probabilities
    semantic_classes = semantic_pred.argmax(dim=0)  # (H, W)

    # Create refined prediction starting from original
    refined_classes = semantic_classes.clone()

    # Track which pixels have been assigned to an instance
    # assigned_mask = torch.zeros((H, W), dtype=torch.bool, device=device)

    # Process each instance mask
    for mask_dict in instance_masks:
        instance_mask = mask_dict["segmentation"]

        # Convert to torch tensor if needed
        if isinstance(instance_mask, np.ndarray):
            instance_mask = torch.from_numpy(instance_mask).to(device)
        elif not isinstance(instance_mask, torch.Tensor):
            continue

        # Ensure mask is boolean
        if instance_mask.dtype != torch.bool:
            instance_mask = instance_mask.bool()

        # Find pixels in this instance
        instance_pixels = instance_mask  # & ~assigned_mask

        if not instance_pixels.any():
            continue

        # Get semantic predictions within this instance
        instance_semantic_classes = semantic_classes[instance_pixels]

        # Find majority class (mode)
        if len(instance_semantic_classes) == 0:
            continue

        # Use bincount for efficient majority voting
        majority_class = torch.bincount(instance_semantic_classes.flatten()).argmax()

        # Assign majority class to all pixels in this instance
        refined_classes[instance_pixels] = majority_class

        # Mark these pixels as assigned
        # assigned_mask |= instance_mask

    # Convert refined classes back to one-hot/probability format
    # Create one-hot encoding
    refined_pred = torch.zeros_like(semantic_pred)
    refined_pred.scatter_(0, refined_classes.unsqueeze(0), 1.0)

    return refined_pred
