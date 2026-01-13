"""
Dataset for the Stanford-2D-3D Semantic dataset

Author: Mahdi Chamseddine
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from panosamic.datasets.augmentations import Augmentation
from panosamic.datasets.base import BaseDataset


class Stanford2d3dsDataset(BaseDataset):
    ALL_AREAS = ("area_1", "area_2", "area_3", "area_4", "area_5a", "area_5b", "area_6")
    # fmt: off
    CLASS_NAMES = (
        "beam", "board", "bookcase", "ceiling", "chair", "clutter",
        "column", "door", "floor", "sofa", "table", "wall", "window"
        # fmt: on
    )
    NUM_CLASSES = len(CLASS_NAMES)

    def __init__(
        self,
        *,
        dataset_path: Path,
        fold_n: int = 1,
        eval_mode: bool = False,
        mask_black=False,
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
            cross_validation=True,
        )

        self.input_areas = self._get_split(fold_n)
        self.depth_threshold = self._get_depth_threshold(fold_n, depth_inlier_ratio)

        self.sample_list = self._generate_sample_list()
        self.instance_labels, _ = load_semantic_labels(data_path=dataset_path)
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
        instances = Image.open(sample_path / "instances.webp").convert("RGB")

        # Processing
        # Depth is used as a grayscale image
        depth_m = np.array(depth, dtype=int) * 128 / (256 * 256 - 1)
        depth_m = np.clip(depth_m, 0, self.depth_threshold)
        depth = 1 - (depth_m / self.depth_threshold)
        depth *= np.array(depth_mask, bool)[..., -1]
        depth = np.repeat(depth[:, :, np.newaxis], 3, axis=2) * 255

        # Normals are used as an image no need to normalize them
        # normals = (normals - 127.5) / 127.5  # Normals are centered at 127.5

        # Based on HoHoNet data preparation
        instances = np.array(instances, dtype=int)
        unk = instances[..., 0] != 0
        # semantics = instance_labels[
        #     instances[..., 0] * 256 * 256
        #     + instances[..., 1] * 256
        #     + instances[..., 2]
        #     # only two bytes are enough since total number is ~10K < 65535
        # ]
        instances = instances[..., 1] * 256 + instances[..., 2]
        instances[unk] = 0

        semantics = self.instance_labels[instances]
        semantics = semantics.astype(int)

        return (
            np.array(image),
            depth,
            np.array(normals),
            instances - 1,  # -1 to force <UNK> class to ignore_index
            semantics - 1,  # -1 to force <UNK> class to ignore_index
        )

    def _get_split(self, fold_n: int) -> list[str]:
        match fold_n:
            case 1:
                train_areas = [0, 1, 2, 3, 6]
                test_areas = [4, 5]
            case 2:
                train_areas = [0, 2, 4, 5, 6]
                test_areas = [1, 3]
            case 3:
                train_areas = [1, 3, 4, 5]
                test_areas = [0, 2, 6]
            case _:
                raise ValueError("The fold number must be one of [1, 2, 3]")

        if self.eval_mode:
            return [self.ALL_AREAS[i] for i in test_areas]
        else:
            return [self.ALL_AREAS[i] for i in train_areas]


def load_semantic_labels(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    # Based on the code in data preparation for HoHoNet
    # https://github.com/sunset1995/HoHoNet/blob/master/README_prepare_data_s2d3d.md

    # Load semantic classes of all instances
    path = data_path / "assets" / "semantic_labels.json"
    with open(path, "r") as f:
        instance_classes = [name.split("_")[0] for name in json.load(f)] + ["<UNK>"]

    # Load numerical label of semantic classes
    path = data_path / "assets" / "name2label.json"
    with open(path, "r") as f:
        class_labels = json.load(f)

    # Load colors
    path = data_path / "assets" / "colors.npy"
    colors = np.load(str(path.absolute()))

    # Map semantic class to numerical label of each instance  # From HoHoNet
    instance_labels = np.array(
        [class_labels[instance] for instance in instance_classes], np.uint8
    )

    return instance_labels, colors


def main():
    data_path = Path("/data/Datasets/Stanford2D3DS/Stanford2D3D_noXYZ/processed")
    dataset = Stanford2d3dsDataset(
        dataset_path=data_path,
        eval_mode=False,
        fold_n=1,
    )
    print(f"Dataset length: {len(dataset)}")

    Path("temp").mkdir(parents=True, exist_ok=True)
    for input_dict, label_dict in dataset:
        if np.random.rand() > 0.05:
            continue
        print("\n".join(str(v.shape) for v in input_dict.values()))

        image = input_dict["image"].numpy().astype(np.uint8).transpose((1, 2, 0))
        depth = input_dict["depth"].numpy().astype(np.uint8).transpose((1, 2, 0))
        normals = input_dict["normals"].numpy().astype(np.uint8).transpose((1, 2, 0))
        Image.fromarray(image, mode="RGB").save(
            "temp/test_image.webp", format="webp", lossless=True
        )
        Image.fromarray(depth, mode="RGB").save(
            "temp/test_depth.webp", format="webp", lossless=True
        )
        Image.fromarray(normals, mode="RGB").save(
            "temp/test_normals.webp", format="webp", lossless=True
        )

        return


if __name__ == "__main__":
    main()
