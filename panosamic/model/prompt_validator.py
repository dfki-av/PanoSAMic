"""
Author: Mahdi Chamseddine
"""

from typing import Any

import numpy as np
import torch


def prompt_validator(
    prompt: dict[str, Any],
    points_per_side: int,
    device: Any,
    image_size: int = 1024,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    # If prompt is not given or prompt is unclear then segment everything
    if (
        prompt is None
        or not isinstance(prompt, dict)
        or not ({"point_coords", "point_labels"} <= prompt.keys())
    ):
        # Generate points grid for segmenting everything
        grid_point_coords, grid_point_labels = build_points_grid(
            points_per_side, device
        )

        # Scale points to image dimensions
        grid_point_coords *= image_size

        # Scale point coordinates from [0,1] to image dimensions
        point_coords, point_labels = (
            grid_point_coords[:, None, :],
            grid_point_labels[:, None],
        )

        prompt = {"point_coords": point_coords, "point_labels": point_labels}

    return prompt, prompt["point_coords"], prompt["point_labels"]


# Slightly modified version of the build_point_grid from SAM
# from segment_anything.utils.amg import build_point_grid
def build_points_grid(
    points_per_side: int, device: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1], scaled to image size."""
    offset = 1 / (2 * points_per_side)
    points_one_side = np.linspace(offset, 1 - offset, points_per_side)
    points_x = np.tile(points_one_side[None, :], (points_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, points_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)

    # Create tensors with correct device
    point_coords = torch.as_tensor(points, dtype=torch.float, device=device)
    point_labels = torch.ones(points.shape[0], dtype=torch.int, device=device)
    return point_coords, point_labels
