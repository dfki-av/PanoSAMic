"""SAM3 output-semantics tests (CUDA)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests.sam3.helpers import make_inputs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

_H, _W = 256, 256


def _run(model, processor):
    inputs = make_inputs(processor, "cuda")
    with torch.no_grad():
        return model(**inputs)


def test_combined_scores_in_unit_interval(sam3_cuda_model, sam3_processor):
    out = _run(sam3_cuda_model, sam3_processor)
    presence = out.presence_logits[0, 0].float().sigmoid()
    scores = out.pred_logits[0].float().sigmoid() * presence
    assert (scores >= 0).all() and (scores <= 1).all()


def test_sem_pred_values_in_unit_interval(sam3_cuda_model, sam3_processor):
    out = _run(sam3_cuda_model, sam3_processor)
    presence = out.presence_logits[0, 0].float().sigmoid()
    scores = out.pred_logits[0].float().sigmoid() * presence
    keep = scores > 0.0
    if keep.any():
        masks_logits = out.pred_masks[0][keep].float()
        masks_up = F.interpolate(
            masks_logits.unsqueeze(1),
            size=(_H, _W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        sem_channel = (masks_up.sigmoid() * scores[keep].view(-1, 1, 1)).amax(dim=0)
        assert (sem_channel >= 0).all() and (sem_channel <= 1).all()
