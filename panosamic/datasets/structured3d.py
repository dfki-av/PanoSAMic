"""
Dataset for the Structured-3D Semantic dataset

Author: Mahdi Chamseddine
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from panosamic.datasets.augmentations import Augmentation
from panosamic.datasets.base import BaseDataset


class Structured3dDataset(BaseDataset):
    ALL_AREAS = tuple(f"scene_{i:05d}" for i in range(3500))
    # fmt: off
    CLASS_NAMES = (
        "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window",
        "bookshelf", "picture", "counter", "blinds", "desk", "shelves", "curtain",
        "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling", "books",
        "refrigerator", "television", "paper", "towel", "shower curtain", "box",
        "whiteboard", "person", "nightstand", "toilet", "sink", "lamp", "bathtub", "bag",
        "otherstructure", "otherfurniture", "otherprop",
        # fmt: on
    )
    NUM_CLASSES = len(CLASS_NAMES)

    def __init__(
        self,
        *,
        dataset_path: Path,
        fold_n: int = 1,
        eval_mode: bool = False,
        depth_inlier_ratio: float = 0.995,
        ignore_index: int = -1,
        semantic_only: bool = False,
        augmentations: tuple[Augmentation, ...] = (),
        compute_weights: bool = False,
        # Number of times and list of classes to oversample
        oversample: tuple[int, list[str]] = (0, []),
    ):
        super().__init__(
            dataset_path=dataset_path,
            eval_mode=eval_mode,
            mask_black=True,  # Always mask_black areas
            depth_inlier_ratio=depth_inlier_ratio,
            ignore_index=ignore_index,
            augmentations=augmentations,
            cross_validation=False,
        )

        self.input_areas = self._get_split(fold_n)
        self.depth_threshold = self._get_depth_threshold(fold_n, depth_inlier_ratio)
        self.semantic_only = semantic_only
        self.augmentations = augmentations
        self.enable_augmentation = len(augmentations) > 0
        self.augmentation_probabilities = None

        self.sample_list = self._generate_sample_list()
        self.class_weights = (
            self._get_class_weights(fold_n=fold_n, oversample=oversample)
            if compute_weights
            else None
        )

    def _load_sample(
        self, idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sample_path = self.sample_list[idx]
        # Loading
        image = Image.open(sample_path / "rgb.webp").convert("RGB")
        depth = Image.open(sample_path / "depth.png")
        depth_mask = Image.open(sample_path / "depth_mask.webp")
        normals = Image.open(sample_path / "normals.webp").convert("RGB")
        semantics = Image.open(sample_path / "semantics.png")

        # Processing
        # Depth is used as a grayscale image
        depth_m = np.array(depth, int) / 1000  # depth is in mm
        depth_m = np.clip(depth_m, 0, self.depth_threshold)
        depth = 1 - (depth_m / self.depth_threshold)
        depth *= np.array(depth_mask, bool)[..., -1]
        depth = np.repeat(depth[:, :, np.newaxis], 3, axis=2) * 255

        # Normals are used as an image no need to normalize them

        return (
            np.array(image),
            depth,
            np.array(normals),
            # -1 to force background class to ignore_index
            np.array([]),  # instances - 1,
            np.array(semantics, dtype=int) - 1,
        )

    def _get_split(self, fold_n: int) -> list[str]:
        train_scenes = range(3000)
        match fold_n:
            case 1:
                test_scenes = range(3000, 3250)  # Validation set
            case 2:
                test_scenes = range(3250, 3500)  # Test set
            case _:
                raise ValueError("The fold number must be one of [1, 2]")

        if self.eval_mode:
            return [self.ALL_AREAS[i] for i in test_scenes]
        else:
            return [self.ALL_AREAS[i] for i in train_scenes]


def main():
    data_path = Path("/data/Datasets/Structured3D/processed")
    dataset = Structured3dDataset(
        dataset_path=data_path,
        eval_mode=False,
        fold_n=1,
        compute_weights=True,
        semantic_only=True,
    )
    print(f"Dataset length: {len(dataset)}")
    sample_data, sample_labels = dataset[50]
    print(torch.unique(sample_labels["semantics"]))


if __name__ == "__main__":
    main()
