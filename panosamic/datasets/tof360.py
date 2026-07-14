"""
Dataset for the ToF-360 dataset

Author: Mahdi Chamseddine
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from panosamic.datasets.augmentations import Augmentation
from panosamic.datasets.base import BaseDataset
from panosamic.datasets.stanford2d3ds import Stanford2d3dsDataset


class ToF360Dataset(BaseDataset):
    """Real-world ToF-360 panoramas, remapped to Stanford2D3DS's 13-class taxonomy.

    ToF-360 (https://huggingface.co/datasets/COLE-Ricoh/ToF-360) has only 179
    samples across 4 scenes -- too little to train on. This dataset exists to
    zero-shot evaluate a Stanford2D3DS-pretrained checkpoint against real
    (non-synthetic) captures, so ``fold_n=1`` uses all scenes for both train and
    eval mode and ``NUM_CLASSES`` matches Stanford2D3DS's.
    """

    ALL_AREAS = ("Hospital", "Office_Room_1", "Office_Room_2", "Parking_Lot")
    CLASS_NAMES = Stanford2d3dsDataset.CLASS_NAMES
    NUM_CLASSES = len(CLASS_NAMES)

    def __init__(
        self,
        *,
        dataset_path: Path,
        fold_n: int = 1,
        eval_mode: bool = False,
        mask_black: bool = False,
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
            mask_black=mask_black,
            depth_inlier_ratio=depth_inlier_ratio,
            ignore_index=ignore_index,
            semantic_only=semantic_only,
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
        depth_m = np.array(depth, int) / 512  # depth is in 1/512m units
        depth_m = np.clip(depth_m, 0, self.depth_threshold)
        depth = 1 - (depth_m / self.depth_threshold)
        depth *= np.array(depth_mask, bool)[..., -1]
        depth = np.repeat(depth[:, :, np.newaxis], 3, axis=2) * 255

        # Normals are used as an image no need to normalize them

        return (
            np.array(image),
            depth,
            np.array(normals),
            np.array([]),  # instances - not annotated for ToF-360
            # -1 to force background/invalid class to ignore_index
            self._map_semantic_labels(np.array(semantics, dtype=int)) - 1,
        )

    def _get_split(self, fold_n: int) -> list[str]:
        if fold_n != 1:
            raise ValueError(
                "ToF-360 is an eval-only dataset (179 samples across 4 scenes); "
                "only fold_n=1 is supported, using all scenes."
            )
        return list(self.ALL_AREAS)

    def _map_semantic_labels(self, labels: np.ndarray) -> np.ndarray:
        # Raw ToF-360 semantic label id -> Stanford2D3DS class (any id not listed
        # here, e.g. light fixtures/pipes/trash bins/signage, falls through to
        # "clutter"). Mirrors Matterport3dDataset._map_semantic_labels's pattern
        # of a hardcoded id->class dict resolved at load time.
        mapping = {
            0: 0,  # "unlabeled" -> void
            33: 0,  # unlabeled variant -> void
            34: 0,  # "invalid" -> void
            1: 12,  # "wall"
            40: 12,  # "curtain wall" -> wall
            2: 4,  # "ceiling"
            20: 4,  # ceiling variant
            42: 4,  # ceiling variant
            3: 9,  # "floor"
            18: 9,  # floor variant
            4: 10,  # "couch" -> sofa
            5: 5,  # "chair"
            7: 8,  # "door"
            8: 13,  # "window"
            12: 11,  # "table"
            14: 7,  # "column"
            17: 1,  # "beam"
            26: 2,  # "board"
            31: 3,  # "shelf" -> bookcase
        }
        clutter_label = self.CLASS_NAMES.index("clutter") + 1
        mapper = np.vectorize(lambda label: mapping.get(label, clutter_label))
        return mapper(labels)


def main():
    data_path = Path("/data/Datasets/ToF360/processed")
    dataset = ToF360Dataset(
        dataset_path=data_path,
        eval_mode=True,
        fold_n=1,
        compute_weights=False,
        semantic_only=True,
    )
    print(f"Dataset length: {len(dataset)}")
    _sample_data, sample_labels = dataset[0]
    print(torch.unique(sample_labels["semantics"]))


if __name__ == "__main__":
    main()
