"""Shared pytest fixtures for the PanoSAMic test suite."""

import pytest

from panosamic.model.model_builder import panosamic_builder
from tests._helpers import _BASELINE_CFG, _FULL_CFG, NUM_CLASSES


@pytest.fixture(scope="module")
def baseline_model():
    return panosamic_builder(
        _BASELINE_CFG, num_classes=NUM_CLASSES, freeze_encoder=True
    )


@pytest.fixture(scope="module")
def full_model():
    return panosamic_builder(_FULL_CFG, num_classes=NUM_CLASSES, freeze_encoder=True)
