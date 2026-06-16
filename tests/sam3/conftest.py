"""Session-scoped SAM3 fixtures.

All fixtures call pytest.importorskip so the entire sam3/ test suite is
automatically skipped when the optional extra is not installed:

    uv sync --extra sam3
"""

from __future__ import annotations

import pytest
import torch
from tqdm.auto import tqdm

_MODEL_ID = "facebook/sam3"


@pytest.fixture(scope="session")
def sam3_processor():
    Sam3Processor = pytest.importorskip(
        "transformers",
        reason="install with: uv sync --extra sam3",
    ).Sam3Processor
    with tqdm(desc="Loading SAM3 processor", total=1, unit="proc") as pbar:
        proc = Sam3Processor.from_pretrained(_MODEL_ID)
        pbar.update(1)
    return proc


@pytest.fixture(scope="session")
def sam3_base_model():
    """SAM3 in float32 on CPU — portable across CPU and MPS."""
    Sam3Model = pytest.importorskip(
        "transformers",
        reason="install with: uv sync --extra sam3",
    ).Sam3Model
    with tqdm(desc="Loading SAM3 weights", total=1, unit="model") as pbar:
        model = Sam3Model.from_pretrained(_MODEL_ID, torch_dtype=torch.float32)
        pbar.update(1)
    model.eval()
    return model


@pytest.fixture(scope="session")
def sam3_cuda_model():
    """SAM3 in bfloat16 on CUDA — separate from sam3_base_model to avoid OOM on shared GPUs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    Sam3Model = pytest.importorskip(
        "transformers",
        reason="install with: uv sync --extra sam3",
    ).Sam3Model
    with tqdm(
        desc="Loading SAM3 weights (CUDA/bfloat16)", total=1, unit="model"
    ) as pbar:
        model = Sam3Model.from_pretrained(_MODEL_ID, torch_dtype=torch.bfloat16)
        pbar.update(1)
    model.to("cuda")
    model.eval()
    return model
