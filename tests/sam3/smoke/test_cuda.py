"""SAM3 CUDA smoke tests."""

from __future__ import annotations

import pytest
import torch

from tests.sam3.helpers import assert_sam3_output, make_inputs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def test_forward_shapes(sam3_cuda_model, sam3_processor):
    inputs = make_inputs(sam3_processor, "cuda")
    with torch.no_grad():
        out = sam3_cuda_model(**inputs)
    assert_sam3_output(out, "cuda")


def test_outputs_on_cuda(sam3_cuda_model, sam3_processor):
    inputs = make_inputs(sam3_processor, "cuda")
    with torch.no_grad():
        out = sam3_cuda_model(**inputs)
    assert out.pred_masks.device.type == "cuda"
    assert out.pred_boxes.device.type == "cuda"
    assert out.pred_logits.device.type == "cuda"
    assert out.presence_logits.device.type == "cuda"
