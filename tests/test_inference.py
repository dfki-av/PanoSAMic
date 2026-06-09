"""
Offline inference smoke tests — no datasets, no GPU, no SAM weights needed.

Tests cover:
  - Model construction (both baseline and full configs)
  - trainable_state_dict() key correctness
  - save_pretrained / reload cycle (safetensors round-trip)
  - forward() output shape with random inputs
"""

import tempfile
from pathlib import Path

import torch

from panosamic.model.panosamic_net import _FROZEN_PREFIXES
from tests._helpers import _BASELINE_CFG, _FULL_CFG, NUM_CLASSES, make_batch

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_baseline_builds(baseline_model):
    assert baseline_model is not None


def test_full_model_builds(full_model):
    assert full_model is not None


# ---------------------------------------------------------------------------
# trainable_state_dict
# ---------------------------------------------------------------------------


def test_trainable_state_dict_excludes_frozen(baseline_model):
    td = baseline_model.trainable_state_dict()
    frozen = [k for k in td if k.startswith(_FROZEN_PREFIXES)]
    assert not frozen, f"Frozen keys leaked into trainable_state_dict: {frozen}"


def test_trainable_state_dict_baseline_has_decoder(baseline_model):
    # Baseline uses BaselineDecoder; PanoSAMic sets feature_fuser=None in that path,
    # so only semantic_decoder.* is trainable.
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


# ---------------------------------------------------------------------------
# Save / reload cycle (no forward pass, no SAM weights)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# load_sam_backbone — semantic_only compatibility
# ---------------------------------------------------------------------------


def test_load_sam_backbone_tolerates_prompt_and_mask_decoder_keys(baseline_model):
    # semantic_only=True models have no prompt_encoder / mask_decoder submodules.
    # load_sam_backbone must not raise when the SAM .pth contains those keys.
    from panosamic.model.model_builder import load_sam_backbone

    # Build a mock SAM state dict: real image_encoder keys + stub prompt/mask keys
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

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pth") as f:
        torch.save(mock_state, f.name)
        load_sam_backbone(baseline_model, Path(f.name))  # must not raise


# ---------------------------------------------------------------------------
# Forward pass — output shape
# ---------------------------------------------------------------------------


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
