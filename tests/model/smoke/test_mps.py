"""PanoSAMic MPS forward-pass smoke tests."""

from __future__ import annotations

import pytest
import torch

from tests._helpers import _BASELINE_CFG, _FULL_CFG, NUM_CLASSES, make_batch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)


@pytest.fixture(scope="module")
def baseline_model_mps(baseline_model):
    return baseline_model.to("mps")


@pytest.fixture(scope="module")
def full_model_mps(full_model):
    return full_model.to("mps")


@torch.no_grad()
def test_mps_forward_baseline(baseline_model_mps):
    out = baseline_model_mps(make_batch(_BASELINE_CFG.modalities, device="mps"))
    assert len(out) == 1
    sem = out[0]["sem_preds"]
    assert sem.device.type == "mps"
    assert sem.shape[1] == NUM_CLASSES


@torch.no_grad()
def test_mps_forward_full_model(full_model_mps):
    out = full_model_mps(make_batch(_FULL_CFG.modalities, device="mps"))
    assert len(out) == 1
    sem = out[0]["sem_preds"]
    assert sem.device.type == "mps"
    assert sem.shape[1] == NUM_CLASSES


@torch.no_grad()
def test_mps_forward_batch_size_two(baseline_model_mps):
    out = baseline_model_mps(make_batch(_BASELINE_CFG.modalities, device="mps") * 2)
    assert len(out) == 2
    for item in out:
        assert item["sem_preds"].device.type == "mps"
        assert item["sem_preds"].shape[1] == NUM_CLASSES
