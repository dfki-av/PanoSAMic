"""Tests for panosamic/model/fusion/basic_fusion.py."""

from __future__ import annotations

import torch

from panosamic.model.fusion.basic_fusion import BasicFusion

# ── Fixtures ─────────────────────────────────────────────────────────────────

_N = 1  # batch size
_MOD = 2  # modalities
_C = 4  # channels
_IN = 8  # spatial size in
_OUT = 16  # spatial size out
_DEPTH = 2  # number of encoder branches


def _inputs() -> list[torch.Tensor]:
    """List of `_DEPTH` branch tensors, each (N*MOD, C, IN, IN)."""
    return [torch.randn(_N * _MOD, _C, _IN, _IN) for _ in range(_DEPTH)]


# ── Output shapes ─────────────────────────────────────────────────────────────


def test_concat_output_shape():
    fusion = BasicFusion("concat", _C, _MOD, _IN, _OUT, depth=_DEPTH)
    out = fusion(_inputs())
    assert out.shape == (_N, _DEPTH * _MOD * _C, _OUT, _OUT)


def test_add_output_shape():
    fusion = BasicFusion("add", _C, _MOD, _IN, _OUT, depth=_DEPTH)
    out = fusion(_inputs())
    assert out.shape == (_N, _DEPTH * _C, _OUT, _OUT)


def test_mult_output_shape():
    fusion = BasicFusion("mult", _C, _MOD, _IN, _OUT, depth=_DEPTH)
    out = fusion(_inputs())
    assert out.shape == (_N, _DEPTH * _C, _OUT, _OUT)


# ── Finite values ─────────────────────────────────────────────────────────────


def test_concat_output_finite():
    fusion = BasicFusion("concat", _C, _MOD, _IN, _OUT, depth=_DEPTH)
    assert fusion(_inputs()).isfinite().all()


def test_add_output_finite():
    fusion = BasicFusion("add", _C, _MOD, _IN, _OUT, depth=_DEPTH)
    assert fusion(_inputs()).isfinite().all()


def test_mult_output_finite():
    fusion = BasicFusion("mult", _C, _MOD, _IN, _OUT, depth=_DEPTH)
    assert fusion(_inputs()).isfinite().all()


# ── Fusion semantics ──────────────────────────────────────────────────────────


def test_add_fusion_output_nonzero_for_random_input():
    """Add fusion produces non-zero output for random (non-constant) input."""
    torch.manual_seed(0)
    fusion = BasicFusion("add", _C, _MOD, _IN, _OUT, depth=1)
    branch = torch.randn(1 * _MOD, _C, _IN, _IN)
    out = fusion([branch])
    assert out.isfinite().all()
    assert out.abs().sum() > 0


def test_mult_fusion_zeros_when_one_modality_is_zero():
    """If one modality is all-zero, multiply output must also be zero."""
    fusion = BasicFusion("mult", _C, _MOD, _IN, _OUT, depth=1)
    branch = torch.ones(_N * _MOD, _C, _IN, _IN)
    branch[:1] = 0.0  # zero out first modality
    out = fusion([branch])
    # After LayerNorm the zero modality stays ~0; product should be ~0.
    # Upsampling with bilinear keeps near-zero; check max is tiny.
    assert out.abs().max().item() < 1e-3


def test_out_channels_attribute_concat():
    fusion = BasicFusion("concat", _C, _MOD, _IN, _OUT)
    assert fusion.out_channels == _MOD * _C


def test_out_channels_attribute_add():
    fusion = BasicFusion("add", _C, _MOD, _IN, _OUT)
    assert fusion.out_channels == _C
