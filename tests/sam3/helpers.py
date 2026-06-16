"""Shared helpers for the SAM3 test suite."""

from __future__ import annotations

import numpy as np
from PIL import Image

_rng = np.random.default_rng(0)
DUMMY_IMAGE = Image.fromarray(_rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
DUMMY_PROMPT = "wall"


def make_inputs(processor, device: str) -> dict:
    inputs = processor(images=DUMMY_IMAGE, text=DUMMY_PROMPT, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def assert_sam3_output(outputs, device: str) -> None:
    dev = device.split(":")[0]  # "cuda:0" → "cuda"

    assert outputs.pred_masks is not None
    assert outputs.pred_boxes is not None
    assert outputs.pred_logits is not None
    assert outputs.presence_logits is not None

    assert outputs.pred_masks.ndim == 4, "pred_masks: expected (batch, queries, H, W)"
    assert outputs.pred_boxes.ndim == 3, "pred_boxes: expected (batch, queries, 4)"
    assert outputs.pred_boxes.shape[-1] == 4
    assert outputs.pred_logits.ndim == 2, "pred_logits: expected (batch, queries)"
    assert outputs.presence_logits.shape == (1, 1)

    assert outputs.pred_masks.device.type == dev
    assert outputs.pred_boxes.device.type == dev
    assert outputs.pred_logits.device.type == dev
    assert outputs.presence_logits.device.type == dev
