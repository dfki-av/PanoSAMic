"""Shared constants and input-generation helpers for the test suite."""

from typing import Any

import torch

from panosamic.evaluation.utils.config import ModelConfig

_BASELINE_CFG = ModelConfig(
    vit_model="vit_b",
    modalities=("image",),
    semantic_only=True,
    channel_attention=None,
    spatial_attention=None,
    dual_view_fusion=False,
    basic_fusion="concat",
)

_FULL_CFG = ModelConfig(
    vit_model="vit_b",
    modalities=("image", "depth"),
    semantic_only=True,
    channel_attention={
        "moving_attention": True,
        "window_size": 8,
        "stride": 4,
        "aggregation": "none",
    },
    spatial_attention=None,
    dual_view_fusion=False,
    basic_fusion=None,
)

NUM_CLASSES = 13


def make_batch(
    modalities: tuple[str, ...],
    h: int = 128,
    w: int = 256,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Random input in the format PanoSAMic.forward() expects."""
    sample: dict[str, torch.Tensor] = {}
    for mod in modalities:
        sample[mod] = torch.rand(3, h, w, device=device)
    return [sample]
