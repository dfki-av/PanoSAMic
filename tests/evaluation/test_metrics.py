"""Tests for panosamic/evaluation/metrics.py."""

from __future__ import annotations

import torch

from panosamic.evaluation.metrics import intersection_and_union_gpu

# ── Basic correctness ────────────────────────────────────────────────────────


def test_perfect_prediction():
    output = torch.tensor([0, 1, 2])
    target = torch.tensor([0, 1, 2])
    inter, union, _ = intersection_and_union_gpu(output, target, num_classes=3)
    assert inter.tolist() == [1.0, 1.0, 1.0]
    assert union.tolist() == [1.0, 1.0, 1.0]


def test_partial_overlap():
    output = torch.tensor([0, 0, 1])
    target = torch.tensor([0, 1, 1])
    inter, union, _ = intersection_and_union_gpu(output, target, num_classes=2)
    assert inter[0].item() == 1.0
    assert inter[1].item() == 1.0
    assert union[0].item() == 2.0
    assert union[1].item() == 2.0


def test_no_overlap():
    output = torch.tensor([0, 0])
    target = torch.tensor([1, 1])
    inter, union, _ = intersection_and_union_gpu(output, target, num_classes=2)
    assert inter[0].item() == 0.0
    assert inter[1].item() == 0.0
    assert union[0].item() == 2.0
    assert union[1].item() == 2.0


def test_output_shape():
    output = torch.zeros(4, 4, dtype=torch.long)
    target = torch.zeros(4, 4, dtype=torch.long)
    inter, union, area = intersection_and_union_gpu(output, target, num_classes=5)
    assert inter.shape == (5,)
    assert union.shape == (5,)
    assert area.shape == (5,)


def test_2d_input():
    output = torch.tensor([[0, 1], [2, 0]])
    target = torch.tensor([[0, 1], [0, 2]])
    inter, _, _ = intersection_and_union_gpu(output, target, num_classes=3)
    assert inter[0].item() == 1.0
    assert inter[1].item() == 1.0
    assert inter[2].item() == 0.0


# ── ignore_index ─────────────────────────────────────────────────────────────


def test_ignore_index_excluded_from_counts():
    output = torch.tensor([0, 1, 2])
    target = torch.tensor([0, -1, 2])
    inter, _, area = intersection_and_union_gpu(
        output, target, num_classes=3, ignore_index=-1
    )
    assert inter[1].item() == 0.0
    assert area[1].item() == 0.0


def test_ignore_index_does_not_affect_other_classes():
    output = torch.tensor([0, 0, 1])
    target = torch.tensor([0, -1, 1])
    inter, _, _ = intersection_and_union_gpu(
        output, target, num_classes=2, ignore_index=-1
    )
    assert inter[0].item() == 1.0
    assert inter[1].item() == 1.0


# ── Device handling ───────────────────────────────────────────────────────────


def test_returns_on_cpu_when_input_on_cpu():
    output = torch.tensor([0, 1, 2])
    target = torch.tensor([0, 1, 2])
    inter, union, area = intersection_and_union_gpu(output, target, num_classes=3)
    assert inter.device.type == "cpu"
    assert union.device.type == "cpu"
    assert area.device.type == "cpu"


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_single_class():
    output = torch.tensor([0, 0, 0])
    target = torch.tensor([0, 0, 0])
    inter, union, _ = intersection_and_union_gpu(output, target, num_classes=1)
    assert inter[0].item() == 3.0
    assert union[0].item() == 3.0


def test_area_target_matches_gt_counts():
    output = torch.tensor([0, 1, 0, 2])
    target = torch.tensor([0, 0, 1, 2])
    _, _, area = intersection_and_union_gpu(output, target, num_classes=3)
    assert area[0].item() == 2.0
    assert area[1].item() == 1.0
    assert area[2].item() == 1.0
