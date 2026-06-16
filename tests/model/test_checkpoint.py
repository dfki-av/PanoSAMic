"""Checkpoint save/reload tests (safetensors round-trip)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from panosamic.model.panosamic_net import _FROZEN_PREFIXES


def test_save_pretrained_backbone_free(baseline_model):
    with tempfile.TemporaryDirectory() as tmp:
        baseline_model.save_pretrained(tmp)
        sf = Path(tmp) / "model.safetensors"
        assert sf.exists(), "model.safetensors not written"

        from safetensors.torch import load_file

        ckpt = load_file(str(sf), device="cpu")
        frozen = [k for k in ckpt if k.startswith(_FROZEN_PREFIXES)]
        assert not frozen, f"Backbone keys found in saved checkpoint: {frozen}"


def test_save_reload_key_parity(baseline_model):
    with tempfile.TemporaryDirectory() as tmp:
        baseline_model.save_pretrained(tmp)

        from safetensors.torch import load_file

        ckpt = load_file(str(Path(tmp) / "model.safetensors"), device="cpu")
        trainable = baseline_model.trainable_state_dict()
        assert set(ckpt) == set(trainable), (
            f"Key mismatch after reload.\n"
            f"  Missing: {set(trainable) - set(ckpt)}\n"
            f"  Extra:   {set(ckpt) - set(trainable)}"
        )
        for k in trainable:
            assert ckpt[k].shape == trainable[k].shape, f"Shape mismatch for {k}"
