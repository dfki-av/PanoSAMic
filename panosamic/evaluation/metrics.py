"""
Author: Mahdi Chamseddine
"""

from typing import Any

import numpy as np
import torch


def compute_metrics(
    pred_list: list[dict[str, torch.Tensor]],
    label_list: list[dict[str, torch.Tensor]],
    num_classes: int,
    ignore_index: int,
    device: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Initialize accumulators
    area_intersection = torch.zeros(num_classes, device=device)
    area_union = torch.zeros(num_classes, device=device)
    area_target = torch.zeros(num_classes, device=device)

    for pred, label in zip(pred_list, label_list):
        intersection, union, target = intersection_and_union_gpu(
            torch.argmax(pred["sem_preds"], dim=1).squeeze(0),
            label["semantics"].squeeze(0),
            num_classes,
            ignore_index,
        )

        area_intersection += intersection
        area_union += union
        area_target += target

    return area_intersection, area_union, area_target


def intersection_and_union_gpu(
    output: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> tuple[torch.Tensor, ...]:
    """
    (GPU) Compute the intersection and union for segmentation predictions and targets.
    Based on CyCTR implementation with added changes for readability
    https://github.com/YanFangCS/CyCTR-Pytorch/blob/master/util/util.py

    Args:
        output (torch.Tensor): Predicted labels.
        target (torch.Tensor): Ground truth labels.
        num_classes (int): Number of classes (K).
        ignore_index (int, optional): Label to ignore in the calculation. Default is -1.

    Returns:
        tuple: Tensors containing:
            - area_intersection (torch.Tensor): Intersection area for each class. Shape: (K,).
            - area_union (torch.Tensor): Union area for each class. Shape: (K,).
            - area_target (torch.Tensor): Total target area for each class. Shape: (K,).
    """
    # Validate input dimensions
    assert output.dim() in [1, 2, 3], "Output must have 1, 2, or 3 dimensions."
    assert output.shape == target.shape, "Output and target must have the same shape."

    # Flatten tensors to 1D for easier processing
    output = output.view(-1)
    target = target.view(-1)

    # Ignore specified indices in the output
    output[target == ignore_index] = ignore_index

    # Find intersection: pixels where prediction matches the target
    intersection = output[output == target]

    # Calculate histogram counts for intersection, output, and target
    area_intersection = torch.histc(
        intersection, bins=num_classes, min=0, max=num_classes - 1
    )
    area_output = torch.histc(output, bins=num_classes, min=0, max=num_classes - 1)
    area_target = torch.histc(target, bins=num_classes, min=0, max=num_classes - 1)

    # Union is the sum of output and target areas minus the intersection
    area_union = area_output + area_target - area_intersection

    return area_intersection, area_union, area_target


def intersection_and_union_cpu(
    output: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = -1,
    ignore_ratio: float = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (CPU) Compute the intersection and union for segmentation predictions and targets.
    Based on CyCTR implementation with added changes for readability
    https://github.com/YanFangCS/CyCTR-Pytorch/blob/master/util/util.py

    Args:
        output (np.ndarray): Predicted labels.
        target (np.ndarray): Ground truth labels.
        num_classes (int): Number of classes (K).
        ignore_index (int, optional): Label to ignore in the calculation. Default is -1.

    Returns:
        tuple: arrays containing:
            - area_intersection (np.ndarray): Intersection area for each class. Shape: (K,).
            - area_union (np.ndarray): Union area for each class. Shape: (K,).
            - area_target (np.ndarray): Total target area for each class. Shape: (K,).
    """
    # Based on CyCTR implementation https://github.com/YanFangCS/CyCTR-Pytorch/blob/master/util/util.py
    # K classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.

    assert output.ndim in [1, 2, 3], "Output must have 1, 2, or 3 dimensions."
    assert output.shape == target.shape, "Output and target must have the same shape."

    mask = create_center_mask(target.shape, ratio=ignore_ratio)
    if ignore_ratio > 0:
        target = target[:, mask]
        output = output[:, mask]
    # Flatten tensors to 1D for easier processing
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)

    # Ignore specified indices in the output
    output[np.where(target == ignore_index)[0]] = ignore_index

    # Find intersection: pixels where prediction matches the target
    intersection = output[np.where(output == target)[0]]

    # Calculate histogram counts for intersection, output, and target
    area_intersection, _ = np.histogram(intersection, bins=np.arange(num_classes + 1))
    area_output, _ = np.histogram(output, bins=np.arange(num_classes + 1))
    area_target, _ = np.histogram(target, bins=np.arange(num_classes + 1))

    # Union is the sum of output and target areas minus the intersection
    area_union = area_output + area_target - area_intersection

    return area_intersection, area_union, area_target


def create_center_mask(shape: tuple[int, int], ratio: float) -> np.ndarray:
    """
    Create a binary mask for an (h, w) image where the central region
    spanning `ratio * w` columns is masked (set to 1) and the rest is 0.

    Parameters:
        h (int): Height of the image.
        w (int): Width of the image.
        ratio (float): Ratio of the width to be masked (0 to 1).

    Returns:
        np.ndarray: A binary mask of shape (h, w).
    """
    h, w = shape
    mask = np.zeros(w, dtype=np.uint8)
    center_w = int(ratio * w)
    start = (w - center_w) // 2
    end = start + center_w
    mask[start:end] = 1
    mask = np.invert(mask)
    return mask


def main():
    # Simulated example tensors
    output = torch.tensor([[0, 1, 2], [2, 1, 0]]).to("cuda")
    target = torch.tensor([[0, 1, 1], [2, 1, 0]]).to("cuda")
    num_classes = 3

    # Compute intersection and union
    intersection, union, target_area = intersection_and_union_gpu(
        output, target, num_classes
    )

    print("Intersection:", intersection)
    print("Union:", union)
    print("Target Area:", target_area)


if __name__ == "__main__":
    main()
