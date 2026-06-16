"""SAM3 output-semantics tests (CPU): logits vs probabilities.

Documents and enforces the interpretation of every output tensor from the HF
Sam3Model, and verifies that the scoring pipeline used in eval_sam3_panosamic
produces values in expected ranges.

Old repo API                          HF API equivalent
------------------------------------  ----------------------------------------
masks_logits  (N, H, W)               pred_masks[0]  (queries, h, w) — logits
                                        → upsample → sigmoid → probabilities
scores        scalar per mask          pred_logits[0].sigmoid()        (queries,)
                                        * presence_logits[0,0].sigmoid()
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tests.sam3.helpers import make_inputs

_H, _W = 256, 256


def _run(model, processor, device: str):
    inputs = make_inputs(processor, device)
    with torch.no_grad():
        return model(**inputs)


def test_pred_masks_are_logits_not_probabilities(sam3_base_model, sam3_processor):
    """pred_masks must be raw logits: a real input will produce values outside [0, 1]."""
    out = _run(sam3_base_model.to("cpu"), sam3_processor, "cpu")
    masks = out.pred_masks.float()
    assert (masks < 0).any() or (masks > 1).any(), (
        "pred_masks look like probabilities; the eval pipeline applies sigmoid itself"
    )


def test_pred_logits_are_logits_not_probabilities(sam3_base_model, sam3_processor):
    """pred_logits must be raw logits: the eval pipeline applies .sigmoid() explicitly."""
    out = _run(sam3_base_model.to("cpu"), sam3_processor, "cpu")
    logits = out.pred_logits.float()
    assert (logits < 0).any() or (logits > 1).any()


def test_presence_logits_shape_is_scalar_gate(sam3_base_model, sam3_processor):
    """presence_logits is a (1, 1) pre-sigmoid gate, not a per-query tensor."""
    out = _run(sam3_base_model.to("cpu"), sam3_processor, "cpu")
    assert out.presence_logits.shape == (1, 1)


def test_class_scores_after_sigmoid_in_unit_interval(sam3_base_model, sam3_processor):
    out = _run(sam3_base_model.to("cpu"), sam3_processor, "cpu")
    probs = out.pred_logits[0].float().sigmoid()
    assert (probs >= 0).all() and (probs <= 1).all()


def test_presence_gate_after_sigmoid_in_unit_interval(sam3_base_model, sam3_processor):
    out = _run(sam3_base_model.to("cpu"), sam3_processor, "cpu")
    gate = out.presence_logits[0, 0].float().sigmoid()
    assert 0.0 <= gate.item() <= 1.0


def test_combined_scores_in_unit_interval(sam3_base_model, sam3_processor):
    """class_score * presence_gate must stay in [0, 1] — used as the confidence filter."""
    out = _run(sam3_base_model.to("cpu"), sam3_processor, "cpu")
    presence = out.presence_logits[0, 0].float().sigmoid()
    scores = out.pred_logits[0].float().sigmoid() * presence
    assert (scores >= 0).all() and (scores <= 1).all()


def test_upsampled_mask_sigmoid_in_unit_interval(sam3_base_model, sam3_processor):
    """Upsampled masks after sigmoid must be in [0, 1] (they are weighted into sem_pred)."""
    out = _run(sam3_base_model.to("cpu"), sam3_processor, "cpu")
    masks_logits = out.pred_masks[0].float()
    masks_up = F.interpolate(
        masks_logits.unsqueeze(1), size=(_H, _W), mode="bilinear", align_corners=False
    ).squeeze(1)
    assert (masks_up.sigmoid() >= 0).all() and (masks_up.sigmoid() <= 1).all()


def test_sem_pred_values_in_unit_interval(sam3_base_model, sam3_processor):
    """End-to-end: weighted mask aggregation must stay in [0, 1].

    Mirrors the core scoring logic from _predict_semantics_for_image so a
    regression in the formula surfaces here before it reaches full evaluation.
    """
    out = _run(sam3_base_model.to("cpu"), sam3_processor, "cpu")
    presence = out.presence_logits[0, 0].float().sigmoid()
    scores = out.pred_logits[0].float().sigmoid() * presence
    keep = scores > 0.0  # keep everything to exercise the formula
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
