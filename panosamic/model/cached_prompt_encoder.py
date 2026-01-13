"""
Caching wrapper for SAM's prompt encoder to avoid redundant computations.

When using default grid prompts (no custom boxes/masks), the prompt encoder
produces identical outputs for each batch. This module caches those outputs
to improve efficiency.

Author: Mahdi Chamseddine
"""

from typing import Any

import torch
from segment_anything.modeling.prompt_encoder import PromptEncoder


class CachedPromptEncoder:
    """
    Transparent wrapper around SAM's PromptEncoder that caches outputs for default grid prompts.

    When batched_prompts is empty (using default grid), the same prompts are generated
    repeatedly. This class automatically caches and reuses prompt encoder outputs.

    Usage: Simply replace prompt_encoder with this wrapper - no API changes needed.
    """

    def __init__(self, prompt_encoder: PromptEncoder):
        self._prompt_encoder = prompt_encoder

        # Cache storage
        self._cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._cache_valid = False

    def __call__(
        self,
        points: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor] | None = None,
        boxes: Any = None,
        masks: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompts with automatic caching for default grid prompts.

        This method has the same signature as PromptEncoder, making it a drop-in replacement.
        Caching is transparently applied when boxes and masks are None.

        Args:
            points: Tuple/List of (point_coords, point_labels) or None
            boxes: Optional boxes (disables caching if provided)
            masks: Optional masks (disables caching if provided)

        Returns:
            Tuple of (sparse_embeddings, dense_embeddings)
        """
        # Only use cache for default prompts (no boxes, no masks)
        use_cache = boxes is None and masks is None

        # Return cached results if available
        if use_cache and self._cache_valid and self._cache is not None:
            return self._cache

        # Compute embeddings
        sparse_embeddings, dense_embeddings = self._prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )

        # Cache results for default prompts
        if use_cache:
            self._cache = (sparse_embeddings, dense_embeddings)
            self._cache_valid = True

        return sparse_embeddings, dense_embeddings

    def invalidate_cache(self):
        """Invalidate the cache (called automatically when needed)."""
        self._cache_valid = False
        self._cache = None

    def get_dense_pe(self):
        """Passthrough to prompt_encoder.get_dense_pe()."""
        return self._prompt_encoder.get_dense_pe()
