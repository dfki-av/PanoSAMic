"""SAM3 CPU smoke tests."""

from __future__ import annotations

import torch

from tests.sam3.helpers import assert_sam3_output, make_inputs


def test_forward_shapes(sam3_base_model, sam3_processor):
    model = sam3_base_model.to("cpu")
    inputs = make_inputs(sam3_processor, "cpu")
    with torch.no_grad():
        out = model(**inputs)
    assert_sam3_output(out, "cpu")


def test_outputs_on_cpu(sam3_base_model, sam3_processor):
    model = sam3_base_model.to("cpu")
    inputs = make_inputs(sam3_processor, "cpu")
    with torch.no_grad():
        out = model(**inputs)
    assert out.pred_masks.device.type == "cpu"
    assert out.pred_boxes.device.type == "cpu"
    assert out.pred_logits.device.type == "cpu"
    assert out.presence_logits.device.type == "cpu"


def test_pred_masks_query_count_matches_pred_logits(sam3_base_model, sam3_processor):
    model = sam3_base_model.to("cpu")
    inputs = make_inputs(sam3_processor, "cpu")
    with torch.no_grad():
        out = model(**inputs)
    assert out.pred_masks.shape[1] == out.pred_logits.shape[1]
