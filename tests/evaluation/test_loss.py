"""Tests for panosamic/evaluation/loss.py."""

from __future__ import annotations

import pytest
import torch

from panosamic.evaluation.loss import FocalLoss, PanoSAMicLoss

_C, _H, _W = 4, 8, 16
_IGNORE = -1


def _preds(n=1):
    return [{"sem_preds": torch.randn(1, _C, _H, _W)} for _ in range(n)]


def _labels(cls=0, n=1):
    return [
        {"semantics": torch.full((1, _H, _W), cls, dtype=torch.long)} for _ in range(n)
    ]


# ── CrossEntropyLoss ─────────────────────────────────────────────────────────


def test_cross_entropy_returns_scalar():
    loss_fn = PanoSAMicLoss(config={"CrossEntropyLoss": 1.0}, ignore_index=_IGNORE)
    assert loss_fn(_preds(), _labels()).ndim == 0


def test_cross_entropy_positive():
    loss_fn = PanoSAMicLoss(config={"CrossEntropyLoss": 1.0}, ignore_index=_IGNORE)
    assert loss_fn(_preds(), _labels()).item() > 0


def test_cross_entropy_perfect_prediction_is_low():
    logits = torch.zeros(1, _C, _H, _W)
    logits[:, 0, :, :] = 100.0
    preds = [{"sem_preds": logits}]
    labels = [{"semantics": torch.zeros(1, _H, _W, dtype=torch.long)}]
    loss_fn = PanoSAMicLoss(config={"CrossEntropyLoss": 1.0}, ignore_index=_IGNORE)
    assert loss_fn(preds, labels).item() < 0.01


def test_cross_entropy_all_ignored_returns_nan():
    """CrossEntropyLoss with all pixels ignored produces nan (0/0 mean), not a crash."""
    labels_all_ignored = [
        {"semantics": torch.full((1, _H, _W), _IGNORE, dtype=torch.long)}
    ]
    loss_fn = PanoSAMicLoss(config={"CrossEntropyLoss": 1.0}, ignore_index=_IGNORE)
    loss = loss_fn(_preds(), labels_all_ignored)
    assert loss.isnan()  # nan is the expected/correct result here


# ── JaccardLoss ──────────────────────────────────────────────────────────────


def test_jaccard_returns_scalar():
    loss_fn = PanoSAMicLoss(config={"JaccardLoss": 1.0}, ignore_index=_IGNORE)
    assert loss_fn(_preds(), _labels()).ndim == 0


def test_jaccard_positive():
    loss_fn = PanoSAMicLoss(config={"JaccardLoss": 1.0}, ignore_index=_IGNORE)
    assert loss_fn(_preds(), _labels()).item() > 0


# ── DiceLoss ─────────────────────────────────────────────────────────────────


def test_dice_returns_scalar():
    loss_fn = PanoSAMicLoss(config={"DiceLoss": 1.0}, ignore_index=_IGNORE)
    assert loss_fn(_preds(), _labels()).ndim == 0


def test_dice_positive():
    loss_fn = PanoSAMicLoss(config={"DiceLoss": 1.0}, ignore_index=_IGNORE)
    assert loss_fn(_preds(), _labels()).item() > 0


# ── FocalLoss ────────────────────────────────────────────────────────────────


def test_focal_returns_scalar():
    loss_fn = PanoSAMicLoss(config={"FocalLoss": 1.0}, ignore_index=_IGNORE)
    assert loss_fn(_preds(), _labels()).ndim == 0


def test_focal_positive():
    loss_fn = PanoSAMicLoss(config={"FocalLoss": 1.0}, ignore_index=_IGNORE)
    assert loss_fn(_preds(), _labels()).item() > 0


def test_focal_loss_standalone_shape():
    focal = FocalLoss(ignore_index=_IGNORE)
    logits = torch.randn(2, _C, _H, _W)
    targets = torch.zeros(2, _H, _W, dtype=torch.long)
    assert focal(logits, targets).ndim == 0


# ── Misconfiguration guard ────────────────────────────────────────────────────


def test_no_loss_configured_raises():
    loss_fn = PanoSAMicLoss(config={}, ignore_index=_IGNORE)
    with pytest.raises(RuntimeError):
        loss_fn(_preds(), _labels())


# ── Scheduled weights ─────────────────────────────────────────────────────────


def test_scheduled_weights_sum_to_one():
    cfg = {
        "CrossEntropyLoss": 1.0,
        "DiceLoss": 1.0,
        "ScheduledLoss": {
            "transition_start_ratio": 0.2,
            "transition_finish_ratio": 0.8,
        },
    }
    loss_fn = PanoSAMicLoss(config=cfg, ignore_index=_IGNORE, total_steps=100)
    for step in [0, 20, 50, 80, 100]:
        w1, w2 = loss_fn.get_scheduled_weights(steps=step)
        assert abs(w1 + w2 - 1.0) < 1e-5, f"weights don't sum to 1 at step {step}"
        assert 0.0 <= w1 <= 1.0
        assert 0.0 <= w2 <= 1.0
