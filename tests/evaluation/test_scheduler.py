"""Tests for panosamic/evaluation/scheduler.py (PanoSAMicLRScheduler)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from panosamic.evaluation.scheduler import PanoSAMicLRScheduler

# ── Helper ────────────────────────────────────────────────────────────────────

_START = 1e-6
_MAX = 1e-3
_WARMUP = 10
_WINDDOWN = 80
_TOTAL = 100


def _make_scheduler(intermediate_lr=None) -> PanoSAMicLRScheduler:
    # Trainer always sets initial lr=1 so that LambdaLR returns absolute LR.
    param = nn.Parameter(torch.zeros(1))
    opt = torch.optim.RAdam([param], lr=1)
    return PanoSAMicLRScheduler(
        optimizer=opt,
        start_lr=_START,
        max_lr=_MAX,
        warm_up_steps=_WARMUP,
        wind_down_step=_WINDDOWN,
        total_steps=_TOTAL,
        intermediate_lr=intermediate_lr,
    )


def _lr_at(sched: PanoSAMicLRScheduler, step: int) -> float:
    return sched.lr_lambda(step)


# ── Warm-up phase ─────────────────────────────────────────────────────────────


def test_warmup_starts_at_start_lr():
    sched = _make_scheduler()
    assert _lr_at(sched, 0) == _START


def test_warmup_ends_near_max_lr():
    sched = _make_scheduler()
    lr = _lr_at(sched, _WARMUP - 1)
    assert lr < _MAX
    assert lr > _START


def test_warmup_is_monotonically_increasing():
    sched = _make_scheduler()
    lrs = [_lr_at(sched, s) for s in range(_WARMUP)]
    assert all(lrs[i] <= lrs[i + 1] for i in range(len(lrs) - 1))


# ── Plateau phase ─────────────────────────────────────────────────────────────


def test_plateau_equals_max_lr():
    sched = _make_scheduler()
    for step in [_WARMUP, _WARMUP + 5, _WINDDOWN - 1]:
        assert _lr_at(sched, step) == _MAX, f"plateau mismatch at step {step}"


# ── Winddown phase ────────────────────────────────────────────────────────────


def test_winddown_starts_at_intermediate_lr():
    sched = _make_scheduler()
    # Without intermediate_lr, it defaults to max_lr.
    assert _lr_at(sched, _WINDDOWN) == pytest.approx(_MAX)


def test_winddown_reaches_zero_at_total_steps():
    sched = _make_scheduler()
    assert _lr_at(sched, _TOTAL) == pytest.approx(0.0)


def test_winddown_is_monotonically_decreasing():
    sched = _make_scheduler()
    lrs = [_lr_at(sched, s) for s in range(_WINDDOWN, _TOTAL + 1)]
    assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))


# ── Custom intermediate_lr ────────────────────────────────────────────────────


def test_intermediate_lr_plateau_decays():
    """When intermediate_lr < max_lr, plateau phase linearly decays."""
    inter = _MAX / 2
    sched = _make_scheduler(intermediate_lr=inter)
    lr_start_plateau = _lr_at(sched, _WARMUP)
    lr_end_plateau = _lr_at(sched, _WINDDOWN - 1)
    assert lr_start_plateau == pytest.approx(_MAX)
    assert lr_end_plateau < _MAX
    assert lr_end_plateau >= inter


def test_winddown_starts_at_custom_intermediate_lr():
    inter = _MAX / 2
    sched = _make_scheduler(intermediate_lr=inter)
    assert _lr_at(sched, _WINDDOWN) == pytest.approx(inter)
