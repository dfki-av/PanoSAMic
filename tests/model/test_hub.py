"""HuggingFace Hub integration tests for PanoSAMic checkpoints.

These tests download real files from ``dfki-av/PanoSAMic`` on the Hub and
are therefore skipped by default to avoid network I/O in regular CI runs.

To run them::

    PANOSAMIC_HUB_TESTS=1 uv run pytest tests/model/test_hub.py -v
"""

from __future__ import annotations

import os

import pytest

from panosamic.model.panosamic_net import _FROZEN_PREFIXES

# ── Guard ─────────────────────────────────────────────────────────────────────

_hub = pytest.mark.skipif(
    not os.getenv("PANOSAMIC_HUB_TESTS"),
    reason="set PANOSAMIC_HUB_TESTS=1 to run Hub download tests",
)

_HUB_REPO = "dfki-av/PanoSAMic"
_SUBFOLDER = "stanford2d3ds-vith-rgbdn-fold1"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hub_ckpt_path(tmp_path_factory):
    """Download model.safetensors from the Hub once per test session."""
    from huggingface_hub import hf_hub_download

    tmp = tmp_path_factory.mktemp("hub")
    path = hf_hub_download(
        repo_id=_HUB_REPO,
        filename="model.safetensors",
        subfolder=_SUBFOLDER,
        local_dir=str(tmp),
    )
    return path


@pytest.fixture(scope="module")
def hub_ckpt(hub_ckpt_path):
    from safetensors.torch import load_file

    return load_file(hub_ckpt_path, device="cpu")


@pytest.fixture(scope="module")
def hub_model():
    """vit_h model matching stanford2d3ds-vith-rgbdn-fold1 — built once per session."""
    from panosamic.datasets.stanford2d3ds import Stanford2d3dsDataset
    from panosamic.evaluation.utils.config import ModelConfig
    from panosamic.model.model_builder import panosamic_builder

    cfg = ModelConfig(
        vit_model="vit_h",
        modalities=("image", "depth", "normals"),
        semantic_only=False,
        channel_attention={
            "moving_attention": True,
            "window_size": 8,
            "stride": 4,
            "aggregation": "max",
        },
        spatial_attention={
            "moving_attention": True,
            "window_size": 8,
            "stride": 4,
            "aggregation": "none",
        },
        dual_view_fusion=True,
        basic_fusion=None,
    )
    return panosamic_builder(cfg, num_classes=Stanford2d3dsDataset.NUM_CLASSES)


# ── Download sanity ───────────────────────────────────────────────────────────


@_hub
def test_hub_checkpoint_file_exists(hub_ckpt_path):
    import os

    assert os.path.exists(hub_ckpt_path)
    assert os.path.getsize(hub_ckpt_path) > 0


# ── Key-space integrity ───────────────────────────────────────────────────────


@_hub
def test_hub_checkpoint_contains_no_backbone_keys(hub_ckpt):
    """Backbone weights must be stripped before upload."""
    leaked = [k for k in hub_ckpt if k.startswith(_FROZEN_PREFIXES)]
    assert not leaked, f"Backbone keys found in Hub checkpoint: {leaked[:5]}"


@_hub
def test_hub_checkpoint_has_feature_fuser_keys(hub_ckpt):
    fuser_keys = [k for k in hub_ckpt if k.startswith("feature_fuser.")]
    assert fuser_keys, "No feature_fuser keys found in Hub checkpoint"


@_hub
def test_hub_checkpoint_has_semantic_decoder_keys(hub_ckpt):
    decoder_keys = [k for k in hub_ckpt if k.startswith("semantic_decoder.")]
    assert decoder_keys, "No semantic_decoder keys found in Hub checkpoint"


@_hub
def test_hub_checkpoint_key_count_is_nonzero(hub_ckpt):
    assert len(hub_ckpt) > 0


# ── Load into model ───────────────────────────────────────────────────────────


@_hub
def test_hub_checkpoint_loads_without_unexpected_keys(hub_model, hub_ckpt):
    """Hub checkpoint keys must be a strict subset of the matching model's state dict.

    Only SAM backbone keys should be missing (intentionally excluded from Hub upload).
    """
    import copy

    model = copy.deepcopy(hub_model)
    missing, unexpected = model.load_state_dict(hub_ckpt, strict=False)

    non_backbone_missing = [k for k in missing if not k.startswith(_FROZEN_PREFIXES)]
    assert not non_backbone_missing, (
        f"Non-backbone keys missing: {non_backbone_missing}"
    )
    assert not unexpected, f"Unexpected keys when loading: {unexpected}"


@_hub
def test_full_checkpoint_loads_sam_plus_hub(hub_model, hub_ckpt):
    """Load SAM backbone then Hub trainable weights; verify no NaN in any parameter.

    SAM vit_h weights (~2.4 GB) are auto-downloaded to ~/.cache/panosamic/sam/ and
    reused on subsequent runs. A forward pass is not run (vit_h on CPU takes minutes).
    """
    import copy

    from panosamic.model.model_builder import get_sam_weights_path, load_sam_backbone

    model = copy.deepcopy(hub_model)

    # Step 1: SAM backbone (auto-downloads on first run, then cached).
    sam_path = get_sam_weights_path(None, "vit_h")
    load_sam_backbone(model, sam_path)

    # Step 2: trainable weights from Hub.
    _, unexpected = model.load_state_dict(hub_ckpt, strict=False)
    assert not unexpected, f"Unexpected keys after full load: {unexpected}"

    # Step 3: all parameters must be finite (NaN would indicate a failed/partial load).
    bad = [n for n, p in model.named_parameters() if not p.isfinite().all()]
    assert not bad, f"NaN/Inf parameters after full load: {bad[:5]}"
