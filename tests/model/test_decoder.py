"""Unit tests for semantic decoder components."""

from __future__ import annotations

import torch

from panosamic.model.semantic_decoder import (
    BaselineDecoder,
    ConvDecoder,
    SphericalConv2D,
)

# ── SphericalConv2D ───────────────────────────────────────────────────────────


def test_spherical_conv_output_shape():
    conv = SphericalConv2D(in_channels=8, out_channels=4, kernel_size=3)
    x = torch.randn(2, 8, 16, 32)
    out = conv(x)
    assert out.shape == (2, 4, 16, 32)


def test_spherical_conv_larger_kernel():
    conv = SphericalConv2D(in_channels=4, out_channels=4, kernel_size=7)
    x = torch.randn(1, 4, 64, 128)
    out = conv(x)
    assert out.shape == (1, 4, 64, 128)


def test_spherical_conv_horizontal_wrap():
    """Left/right edges of the feature map should 'see' each other via circular padding."""
    conv = SphericalConv2D(1, 1, kernel_size=3, bias=False)
    # All-ones kernel: every output pixel sums its 3x3 neighbourhood.
    torch.nn.init.ones_(conv.conv.weight)

    # Signal only at the rightmost column.
    x = torch.zeros(1, 1, 4, 8)
    x[:, :, :, -1] = 1.0

    out = conv(x)
    # With circular padding the kernel centred on column 0 sees col[-1]=col[7].
    # Middle rows (not top/bottom zero-padded rows) must be non-zero.
    assert out[:, :, 1:-1, 0].abs().sum() > 0, (
        "Circular padding not propagating right-edge signal to left column"
    )


def test_spherical_conv_no_vertical_wrap():
    """Top and bottom rows should NOT wrap (zero-padded)."""
    conv = SphericalConv2D(1, 1, kernel_size=3, bias=False)
    torch.nn.init.dirac_(conv.conv.weight)

    x = torch.zeros(1, 1, 4, 8)
    x[:, :, -1, :] = 1.0  # signal only at the bottom row

    out = conv(x)
    # Zero padding means row 0 should NOT receive the bottom signal
    assert out[:, :, 0, :].abs().sum() == 0.0, (
        "Top row received signal from bottom (should be zero-padded)"
    )


# ── ConvDecoder ───────────────────────────────────────────────────────────────


def test_conv_decoder_single_view_output_shape():
    depth, in_ch, num_classes, out_size = 4, 16, 13, 32
    decoder = ConvDecoder(
        in_channels=in_ch,
        num_classes=num_classes,
        out_size=out_size,
        depth=depth,
        dual_view_fusion=False,
    )
    x = torch.randn(2, depth * in_ch, out_size, out_size)
    out = decoder(x)
    assert out.shape == (2, num_classes, out_size, out_size)


def test_conv_decoder_dual_view_output_shape():
    """Dual-view mode expects paired rows (batch must be even) and halves the batch."""
    depth, in_ch, num_classes, out_size = 4, 16, 13, 32
    decoder = ConvDecoder(
        in_channels=in_ch,
        num_classes=num_classes,
        out_size=out_size,
        depth=depth,
        dual_view_fusion=True,
    )
    # batch=4: pairs (0,1) and (2,3)
    x = torch.randn(4, depth * in_ch, out_size, out_size)
    out = decoder(x)
    assert out.shape == (2, num_classes, out_size, out_size)


# ── BaselineDecoder ───────────────────────────────────────────────────────────


def test_baseline_decoder_output_shape():
    in_ch, n_mod, num_classes, out_size = 256, 1, 13, 256
    decoder = BaselineDecoder(
        in_channels=in_ch,
        n_modalities=n_mod,
        num_classes=num_classes,
        out_size=out_size,
    )
    # BaselineDecoder expects (N, n_mod * in_ch, H/4, W/4) → (N, num_classes, H/2, W/2)?
    # Looking at the decoder: two 2x upsample layers → input H/4 → output H
    x = torch.randn(1, n_mod * in_ch, out_size // 4, out_size // 4)
    out = decoder(x)
    assert out.shape == (1, num_classes, out_size, out_size)
