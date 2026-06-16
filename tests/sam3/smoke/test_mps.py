"""SAM3 MPS smoke tests."""

from __future__ import annotations

import pytest
import torch

from tests.sam3.helpers import assert_sam3_output, make_inputs

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)


def test_forward_shapes(sam3_base_model, sam3_processor):
    model = sam3_base_model.to("mps")
    inputs = make_inputs(sam3_processor, "mps")
    with torch.no_grad():
        out = model(**inputs)
    assert_sam3_output(out, "mps")


def test_outputs_on_mps(sam3_base_model, sam3_processor):
    model = sam3_base_model.to("mps")
    inputs = make_inputs(sam3_processor, "mps")
    with torch.no_grad():
        out = model(**inputs)
    assert out.pred_masks.device.type == "mps"
    assert out.pred_boxes.device.type == "mps"
    assert out.pred_logits.device.type == "mps"
    assert out.presence_logits.device.type == "mps"
