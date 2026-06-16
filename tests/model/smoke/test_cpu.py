"""PanoSAMic CPU forward-pass smoke tests."""

from __future__ import annotations

import torch

from tests._helpers import _BASELINE_CFG, _FULL_CFG, NUM_CLASSES, make_batch


@torch.no_grad()
def test_forward_baseline_output_shape(baseline_model):
    out = baseline_model(make_batch(_BASELINE_CFG.modalities))
    assert len(out) == 1
    assert out[0]["sem_preds"].shape[1] == NUM_CLASSES


@torch.no_grad()
def test_forward_full_model_output_shape(full_model):
    out = full_model(make_batch(_FULL_CFG.modalities))
    assert len(out) == 1
    assert out[0]["sem_preds"].shape[1] == NUM_CLASSES


@torch.no_grad()
def test_forward_batch_size_two(baseline_model):
    out = baseline_model(make_batch(_BASELINE_CFG.modalities) * 2)
    assert len(out) == 2
    for item in out:
        assert item["sem_preds"].shape[1] == NUM_CLASSES


@torch.no_grad()
def test_output_on_cpu(baseline_model):
    out = baseline_model(make_batch(_BASELINE_CFG.modalities))
    assert out[0]["sem_preds"].device.type == "cpu"
