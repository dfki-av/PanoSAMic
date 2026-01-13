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


class Matterport3dDataset(BaseDataset):
    # fmt: off
    ALL_AREAS = (
        # Training scenes (0 - 60)
        "17DRP5sb8fy", "1LXtFkjw3qL", "1pXnuDYAj8r", "29hnd4uzFmX", "5LpN3gDmAk7",
        "5q7pvUzZiYa", "759xd9YjKW5", "7y3sRwLe3Va", "82sE5b5pLXE", "8WUmhLawc2A",
        "aayBHfsNo7d", "ac26ZMwG7aT", "B6ByNegPMKs", "b8cTxDM8gDG", "cV4RVeZvu5T",
        "D7N2EKCX4Sj", "e9zR4mvMWw7", "EDJbREhghzL", "GdvgFV5R1Z5", "gTV8FGcVJC9",
        "HxpKQynjfin", "i5noydFURQK", "JeFG25nYj2p", "JF19kD82Mey", "jh4fc5c5qoQ",
        "kEZ7cmS4wCh", "mJXqzFtmKg4", "p5wJjkQkbXX", "Pm6F8kyY3z2", "pRbA3pwrgk9",
        "PuKPg4mmafe", "PX4nDJXEHrG", "qoiz87JEwZ2", "rPc6DW4iMge", "s8pcmisQ38h",
        "S9hNv5qa7GM", "sKLMLpTHeUy", "SN83YJsR3w2", "sT4fr6TAbpF", "ULsKaCPVFJR",
        "uNb9QFRL6hY", "Uxmj2M2itWa", "V2XKFyX4ASd", "VFuaQ6m2Qom", "VVfe2KiqLaN",
        "Vvot9Ly1tCj", "vyrNrziPKCB", "VzqfbhrpDEA", "XcA2TqTSSAj", "2n8kARJN3HM",
        "D7G3Y4RVNrH", "dhjEzFoUFzH", "E9uDoFAP3SH", "gZ6f7yhEvPG", "JmbYfDe2QKZ",
        "r1Q1Z4BcV1o", "r47D5H71a5s", "ur6pFq6Qu1A", "VLzqgDo317F", "YmJkqBEsHnH",
        "ZMojNkEp431",
        # Validation scenes (61 - 67)
        "2azQ1b91cZZ", "8194nk5LbLH", "EU6Fwq7SyZv", "oLBMNvg9in8", "QUCTc6BB5sX",
        "TbHJrupSAjP", "X7HyMhZNoso",
        # Test scenes (68 - 85)
        "2t7WUuJeko7", "5ZKStnWn8Zo", "ARNzJeq3xxb", "fzynW3qQPVF", "jtcxE69GiFV",
        "pa4otMbVnkk", "q9vSo1VnCiC", "rqfALeAoiTq", "UwV83HsGsw3", "wc2JMjhGNzB",
        "WYY7iVyf5p8", "YFuZgdQ5vWj", "yqstnuAEVhm", "YVUC4YcDtcY", "gxdoqLR6rwA",
        "gYvKGZ5eRqb", "RPmz2sHmrrY", "Vt2qJdWjCF2",
        # fmt: on
    )
    # fmt: off
    CLASS_NAMES = (
        "wall", "floor", "chair", "door", "table", "picture", "furniture", "objects",
        "window", "sofa", "bed", "sink", "stairs", "ceiling", "toilet", "mirror",
        "shower", "bathtub", "counter", "shelving",
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
        normals = (
            Image.open(sample_path / "normals.webp").convert("RGB")
            if (sample_path / "normals.webp").exists()
            else []
        )
        semantics = Image.open(sample_path / "semantics.png")

        # Processing
        # Depth is used as a grayscale image
        depth_m = np.array(depth, int) / 4000  # depth is in 0.25mm
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
            self._map_semantic_labels(np.array(semantics, dtype=int)) - 1,
        )

    def _get_split(self, fold_n: int) -> list[str]:
        train_scenes = range(61)
        match fold_n:
            case 1:
                test_scenes = range(61, 68)  # Validation set
            case 2:
                test_scenes = range(68, 86)  # Test set
            case _:
                raise ValueError("The fold number must be one of [1, 2]")

        if self.eval_mode:
            return [self.ALL_AREAS[i] for i in test_scenes]
        else:
            return [self.ALL_AREAS[i] for i in train_scenes]

    def _map_semantic_labels(self, labels: np.ndarray) -> np.ndarray:
        mapping = {
            0: 0,  # "void",
            1: 1,  # "wall",
            2: 2,  # "floor",
            3: 3,  # "chair",
            4: 4,  # "door",
            5: 5,  # "table",
            6: 6,  # "picture",
            7: 7,  # "cabinet", # Furniture
            8: 8,  # "cushion", # Objects
            9: 9,  # "window",
            10: 10,  # "sofa",
            11: 11,  # "bed",
            12: 8,  # "curtain", # Objects
            13: 7,  # "chest_of_drawers",# Furniture
            14: 8,  # "plant", # Objects
            15: 12,  # "sink",
            16: 13,  # "stairs",
            17: 14,  # "ceiling",
            18: 15,  # "toilet",
            19: 3,  # "stool", # Chair
            20: 8,  # "towel", # Objects
            21: 16,  # "mirror",
            22: 8,  # "tv_monitor", # Objects
            23: 17,  # "shower",
            24: 8,  # "column", # Objects
            25: 18,  # "bathtub",
            26: 19,  # "counter",
            27: 8,  # "fireplace", # Objects
            28: 8,  # "lighting", # Objects
            29: 8,  # "beam", # Objects
            30: 8,  # "railing", # Objects
            31: 20,  # "shelving",
            32: 9,  # "blinds", # Window
            33: 7,  # "gym_equipment", # Furniture
            34: 7,  # "seating", # Furniture
            35: 8,  # "board_panel", # Objects
            36: 7,  # "furniture", # Furniture
            37: 8,  # "appliances", # Objects
            38: 8,  # "clothes", # Objects
            39: 8,  # "Objects", # Objects
            40: 8,  # "misc", # Objects
            41: 0,  # "unlabeled",
        }
        mapper = np.vectorize(mapping.get)
        return mapper(labels)


def main():
    data_path = Path("/data/Datasets/Matterport3D/processed")
    dataset = Matterport3dDataset(
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
