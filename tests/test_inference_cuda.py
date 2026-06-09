"""
CUDA inference smoke tests.

All tests in this module are skipped when no GPU is available.
They mirror the CPU tests in test_inference.py but run on CUDA and
additionally assert that output tensors land on the GPU.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from panosamic.model.panosamic_net import _FROZEN_PREFIXES
from tests._helpers import _BASELINE_CFG, _FULL_CFG, NUM_CLASSES, make_batch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baseline_model_cuda(baseline_model):
    return baseline_model.cuda()


@pytest.fixture(scope="module")
def full_model_cuda(full_model):
    return full_model.cuda()


# ---------------------------------------------------------------------------
# Forward pass — output on correct device, correct shape
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_cuda_forward_baseline(baseline_model_cuda):
    out = baseline_model_cuda(make_batch(_BASELINE_CFG.modalities, device="cuda"))
    assert len(out) == 1
    sem = out[0]["sem_preds"]
    assert sem.device.type == "cuda"
    assert sem.shape[1] == NUM_CLASSES


@torch.no_grad()
def test_cuda_forward_full_model(full_model_cuda):
    out = full_model_cuda(make_batch(_FULL_CFG.modalities, device="cuda"))
    assert len(out) == 1
    sem = out[0]["sem_preds"]
    assert sem.device.type == "cuda"
    assert sem.shape[1] == NUM_CLASSES


@torch.no_grad()
def test_cuda_forward_batch_size_two(baseline_model_cuda):
    out = baseline_model_cuda(make_batch(_BASELINE_CFG.modalities, device="cuda") * 2)
    assert len(out) == 2
    for item in out:
        assert item["sem_preds"].device.type == "cuda"
        assert item["sem_preds"].shape[1] == NUM_CLASSES


# ---------------------------------------------------------------------------
# Save / reload — safetensors always written to CPU
# ---------------------------------------------------------------------------


def test_cuda_save_pretrained_backbone_free(baseline_model_cuda):
    with tempfile.TemporaryDirectory() as tmp:
        baseline_model_cuda.save_pretrained(tmp)
        sf = Path(tmp) / "model.safetensors"
        assert sf.exists(), "model.safetensors not written"

        from safetensors.torch import load_file

        ckpt = load_file(str(sf), device="cpu")
        frozen = [k for k in ckpt if k.startswith(_FROZEN_PREFIXES)]
        assert not frozen, f"Backbone keys found in saved checkpoint: {frozen}"


def test_cuda_save_reload_key_parity(baseline_model_cuda):
    with tempfile.TemporaryDirectory() as tmp:
        baseline_model_cuda.save_pretrained(tmp)

        from safetensors.torch import load_file

        ckpt = load_file(str(Path(tmp) / "model.safetensors"), device="cpu")

        trainable = {
            k: v.cpu() for k, v in baseline_model_cuda.trainable_state_dict().items()
        }
        assert set(ckpt) == set(trainable), (
            f"Key mismatch after CUDA reload.\n"
            f"  Missing: {set(trainable) - set(ckpt)}\n"
            f"  Extra:   {set(ckpt) - set(trainable)}"
        )
        for k in trainable:
            assert ckpt[k].shape == trainable[k].shape, f"Shape mismatch for {k}"
