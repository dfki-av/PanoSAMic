"""Tests for panosamic/datasets/augmentations.py."""

from __future__ import annotations

import torch

from panosamic.datasets.augmentations import (
    Augmentation,
    augment_image,
    flip_horizontal,
    rotate_horizontal,
    rotate_horizontal_tensor,
)

_C, _H, _W = 3, 4, 8


def _sample():
    data = {"image": torch.arange(_C * _H * _W, dtype=torch.float).reshape(_C, _H, _W)}
    labels = {"semantics": torch.arange(_H * _W, dtype=torch.long).reshape(1, _H, _W)}
    return data, labels


# ── rotate_horizontal_tensor ──────────────────────────────────────────────────


def test_rotate_shifts_columns():
    x = torch.zeros(1, 1, 4)
    x[0, 0, 0] = 1.0
    out = rotate_horizontal_tensor(x, shift=2)
    assert out[0, 0, 2].item() == 1.0


def test_rotate_full_cycle_is_identity():
    x = torch.randn(3, 4, _W)
    assert torch.allclose(rotate_horizontal_tensor(x, shift=_W), x)


def test_rotate_zero_shift_is_identity():
    x = torch.randn(3, _H, _W)
    assert torch.allclose(rotate_horizontal_tensor(x, shift=0), x)


# ── flip_horizontal ───────────────────────────────────────────────────────────


def test_flip_reverses_columns():
    data, labels = _sample()
    orig_img = data["image"].clone()
    orig_lbl = labels["semantics"].clone()
    data, labels = flip_horizontal(data, labels)
    assert torch.allclose(data["image"], orig_img.flip(-1))
    assert torch.allclose(labels["semantics"], orig_lbl.flip(-1))


def test_flip_preserves_shape():
    data, labels = _sample()
    data, labels = flip_horizontal(data, labels)
    assert data["image"].shape == (_C, _H, _W)
    assert labels["semantics"].shape == (1, _H, _W)


def test_double_flip_is_identity():
    data, labels = _sample()
    orig_img = data["image"].clone()
    data, labels = flip_horizontal(data, labels)
    data, labels = flip_horizontal(data, labels)
    assert torch.allclose(data["image"], orig_img)


# ── rotate_horizontal ─────────────────────────────────────────────────────────


def test_rotate_with_fixed_shift():
    data, labels = _sample()
    orig_img = data["image"].clone()
    data, labels = rotate_horizontal(data, labels, shift=3)
    assert torch.allclose(data["image"], orig_img.roll(3, dims=-1))


def test_rotate_applies_same_shift_to_data_and_labels():
    data, labels = _sample()
    orig_img = data["image"].clone()
    orig_lbl = labels["semantics"].clone()
    data, labels = rotate_horizontal(data, labels, shift=2)
    assert torch.allclose(data["image"], orig_img.roll(2, dims=-1))
    assert torch.allclose(labels["semantics"], orig_lbl.roll(2, dims=-1))


# ── augment_image ─────────────────────────────────────────────────────────────


def test_augment_disabled_returns_unchanged():
    data, labels = _sample()
    orig = data["image"].clone()
    data, labels = augment_image(data, labels, (Augmentation.FLIP,), enabled=False)
    assert torch.allclose(data["image"], orig)


def test_augment_probability_zero_skips():
    data, labels = _sample()
    orig = data["image"].clone()
    # probability=0 means the condition `rand > 0` is always true → skip
    data, labels = augment_image(
        data, labels, (Augmentation.FLIP,), probabilities=(0.0,)
    )
    assert torch.allclose(data["image"], orig)


def test_augment_probability_one_always_applies():
    data, labels = _sample()
    orig = data["image"].clone()
    data, labels = augment_image(
        data, labels, (Augmentation.FLIP,), probabilities=(1.0,)
    )
    assert torch.allclose(data["image"], orig.flip(-1))
