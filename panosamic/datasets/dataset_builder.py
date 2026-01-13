"""
Author: Mahdi Chamseddine
"""

from pathlib import Path

from panosamic.datasets.augmentations import Augmentation
from panosamic.datasets.base import BaseDataset
from panosamic.datasets.matterport3d import Matterport3dDataset
from panosamic.datasets.stanford2d3ds import Stanford2d3dsDataset
from panosamic.datasets.structured3d import Structured3dDataset
from panosamic.evaluation.utils.config import TrainingConfig

DEFAULT_AUGMENTATIONS = (
    Augmentation.FLIP,
    Augmentation.ROTATE,
    Augmentation.PERMUTE,
)


def build_dataset(
    dataset_path: Path | str,
    config: TrainingConfig,
    n_modalities: int,
    test_mode: bool = False,
) -> BaseDataset:
    dataset_path = Path(dataset_path) if isinstance(dataset_path, str) else dataset_path
    if test_mode:
        return test_dataset_builder(
            dataset_path, config=config, n_modalities=n_modalities
        )
    else:
        return train_dataset_builder(
            dataset_path, config=config, n_modalities=n_modalities
        )


def train_dataset_builder(
    data_path: Path, config: TrainingConfig, n_modalities: int
) -> BaseDataset:
    if "stanford2d3ds" == config.dataset_name.lower():
        return Stanford2d3dsDataset(
            dataset_path=data_path,
            fold_n=config.fold,
            mask_black=n_modalities == 1,  # mask_black if only RGB
            semantic_only=True,
            augmentations=DEFAULT_AUGMENTATIONS,
            compute_weights=True,
        )
    elif "structured3d" == config.dataset_name.lower():
        return Structured3dDataset(
            dataset_path=data_path,
            fold_n=config.fold,
            semantic_only=True,
            augmentations=DEFAULT_AUGMENTATIONS,
            compute_weights=True,
        )
    elif "matterport3d" in config.dataset_name.lower():
        return Matterport3dDataset(
            dataset_path=data_path,
            fold_n=config.fold,
            semantic_only=True,
            augmentations=DEFAULT_AUGMENTATIONS,
            compute_weights=True,
        )
    else:
        raise NotImplementedError


def test_dataset_builder(
    data_path: Path, config: TrainingConfig, n_modalities: int
) -> BaseDataset:
    if "stanford2d3ds" == config.dataset_name.lower():
        return Stanford2d3dsDataset(
            dataset_path=data_path,
            fold_n=config.fold,
            eval_mode=True,
            mask_black=n_modalities == 1,  # mask_black if only RGB
            semantic_only=True,
        )
    elif "structured3d" == config.dataset_name.lower():
        return Structured3dDataset(
            dataset_path=data_path,
            fold_n=config.fold,
            eval_mode=True,
            semantic_only=True,
        )
    elif "matterport3d" in config.dataset_name.lower():
        return Matterport3dDataset(
            dataset_path=data_path,
            fold_n=config.fold,
            eval_mode=True,
            semantic_only=True,
        )
    else:
        raise NotImplementedError
