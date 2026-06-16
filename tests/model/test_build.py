"""Model construction and trainable state-dict tests."""

from __future__ import annotations

import torch

from panosamic.model.panosamic_net import _FROZEN_PREFIXES


def test_baseline_builds(baseline_model):
    assert baseline_model is not None


def test_full_model_builds(full_model):
    assert full_model is not None


def test_trainable_state_dict_excludes_frozen(baseline_model):
    td = baseline_model.trainable_state_dict()
    frozen = [k for k in td if k.startswith(_FROZEN_PREFIXES)]
    assert not frozen, f"Frozen keys leaked into trainable_state_dict: {frozen}"


def test_trainable_state_dict_baseline_has_decoder(baseline_model):
    td = baseline_model.trainable_state_dict()
    assert any(k.startswith("semantic_decoder.") for k in td), (
        "semantic_decoder.* missing"
    )


def test_trainable_state_dict_full_has_fuser_and_decoder(full_model):
    td = full_model.trainable_state_dict()
    assert any(k.startswith("feature_fuser.") for k in td), "feature_fuser.* missing"
    assert any(k.startswith("semantic_decoder.") for k in td), (
        "semantic_decoder.* missing"
    )


def test_trainable_plus_frozen_equals_full(baseline_model):
    full_keys = set(baseline_model.state_dict())
    trainable_keys = set(baseline_model.trainable_state_dict())
    frozen_keys = {k for k in full_keys if k.startswith(_FROZEN_PREFIXES)}
    assert trainable_keys | frozen_keys == full_keys


def test_load_sam_backbone_tolerates_prompt_and_mask_decoder_keys(baseline_model):
    import tempfile
    from pathlib import Path

    from panosamic.model.model_builder import load_sam_backbone

    real_keys = {
        k: v.clone()
        for k, v in baseline_model.state_dict().items()
        if k.startswith("image_encoder.")
    }
    stub_keys = {
        "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix": torch.zeros(1),
        "mask_decoder.transformer.norm_final_attn.weight": torch.zeros(1),
    }
    mock_state = {**real_keys, **stub_keys}

    with tempfile.NamedTemporaryFile(suffix=".pth") as f:
        torch.save(mock_state, f.name)
        load_sam_backbone(baseline_model, Path(f.name))
