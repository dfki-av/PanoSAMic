"""
Author: Mahdi Chamseddine
"""

import abc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from panosamic.datasets.augmentations import Augmentation, augment_image


class BaseDataset(Dataset):
    ALL_AREAS: tuple[str, ...]
    CLASS_NAMES: tuple[str, ...]
    NUM_CLASSES: int

    CACHE_FILES = {  # Data cache files to speed up processing
        "file_names": "cache_samples_file_names.json",
        "area_stats": "cache_area_depth_statistics.json",
        "split_stats": "cache_splits_depth_statistics.json",
        "class_weights": "cache_class_weights.json",
    }
    D_INLIER_RATIOS = (0.9, 0.95, 0.99, 0.995, 0.999, 1)  # Used for clipping depth

    def __init__(
        self,
        dataset_path: Path,
        eval_mode: bool,
        mask_black: bool,
        depth_inlier_ratio: float = 0.995,
        ignore_index: int = -1,
        semantic_only: bool = False,
        augmentations: tuple[Augmentation, ...] = (),
        compute_weights: bool = False,
        # Number of times and list of classes to oversample
        oversample: tuple[int, list[str]] = (0, []),
        cross_validation: bool = False,  # Dataset supports cross_validation
    ) -> None:
        super().__init__()

        # Sanity checks
        assert depth_inlier_ratio in self.D_INLIER_RATIOS, (
            f"inlier_ratio must be one of {self.D_INLIER_RATIOS}"
        )

        assert oversample[0] >= 0, f"Oversampling {oversample[0]} must be >= 0"
        if oversample[0] > 0:
            assert compute_weights, "oversampling > 0: set compute_weights to True"
            assert oversample[1], "oversampling > 0: class list can't be empty"

        self.dataset_path = dataset_path
        self.eval_mode = eval_mode
        self.mask_black = mask_black
        self.ignore_index = ignore_index
        self.semantic_only = semantic_only

        self.augmentations = augmentations
        self.enable_augmentation = len(augmentations) > 0
        self.augmentation_probabilities = None

        self.cross_validation = cross_validation

        self.input_areas: list[str]
        self.sample_list: list[Path]

        self.class_weights: torch.Tensor | None

    @abc.abstractmethod
    def _load_sample(
        self, idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_split(self, fold_n: int) -> list[str]:
        raise NotImplementedError

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        image, depth, normals, instances, semantics = self._load_sample(idx)

        if self.mask_black:
            where_black = image.sum(-1) == 0
            semantics[where_black] = self.ignore_index

        # Create tensors from numpy arrays
        image = np.ascontiguousarray(image.transpose((2, 0, 1)), dtype=np.float32)
        image_tensor = torch.as_tensor(image)

        depth = np.ascontiguousarray(depth.transpose((2, 0, 1)), dtype=np.float32)
        depth_tensor = torch.as_tensor(depth)

        if normals.size:
            normals = np.ascontiguousarray(
                normals.transpose((2, 0, 1)), dtype=np.float32
            )
            normals_tensor = torch.as_tensor(normals)
        else:
            normals_tensor = torch.tensor([])

        semantics = np.ascontiguousarray(semantics[np.newaxis, ...], dtype=np.int64)
        semantics_tensor = torch.as_tensor(semantics)

        instances_tensor = torch.tensor([])
        if not self.semantic_only:
            raise NotImplementedError
            if self.mask_black:
                instances[where_black] = self.ignore_index
            instances = np.ascontiguousarray(instances[np.newaxis, ...], dtype=np.int64)
            instances_tensor = torch.as_tensor(instances)

        sample_data = dict(
            image=image_tensor,
            depth=depth_tensor,
            normals=normals_tensor,
        )
        sample_labels = dict(
            semantics=semantics_tensor,
            instances=instances_tensor,
        )

        return augment_image(
            sample_data,
            sample_labels,
            augmentations=self.augmentations,
            enabled=self.enable_augmentation,
            probabilities=self.augmentation_probabilities,
        )

    def _generate_sample_list(self) -> list[Path]:
        sample_list = []
        sample_name_cache = self.dataset_path / self.CACHE_FILES["file_names"]
        area_dict = None
        if sample_name_cache.exists():
            # Cache file found, generating the list from the input file
            with open(sample_name_cache, "r") as f:
                area_dict = json.load(f)

        for area in self.input_areas:
            area_path = self.dataset_path / area
            if (not area_path.exists()) or (not any(area_path.iterdir())):
                # Area/Scene folder is empty or doesn't exist
                continue
            if area_dict:
                sample_list.extend([area_path / frame for frame in area_dict[area]])
            else:
                # No cache file was found, walking the contents of the folder
                sample_list.extend(
                    [frame for frame in area_path.iterdir() if frame.is_dir()]
                )

        assert sample_list, "No files were found, check dataset path."

        return sample_list

    def _get_class_weights(
        self,
        fold_n: int,
        oversample: tuple[int, list[str]] = (0, []),
    ) -> torch.Tensor:
        class_weights_cache = self.dataset_path / self.CACHE_FILES["class_weights"]
        weights, weights_cache = load_cached(
            cache_file=class_weights_cache, fold_n=fold_n
        )

        if weights:
            # Cache hit
            weights = torch.tensor(weights).type(torch.float32)
        else:
            # Cache miss
            weights = self.__compute_class_weights(oversample)

            if self.cross_validation:
                weights_cache[fold_n] = weights.tolist()
            else:
                weights_cache = weights.tolist()

            with open(class_weights_cache, "w") as fp:
                json.dump(weights_cache, fp)

        return weights

    def _get_depth_threshold(self, fold_n: int, inlier_ratio: float) -> float:
        split_depth_cache = self.dataset_path / self.CACHE_FILES["split_stats"]

        depth_thresholds, depth_threshold_cache = load_cached(
            cache_file=split_depth_cache, fold_n=fold_n
        )

        if depth_thresholds:  # Return cached threshold if found
            # Cache hit
            return depth_thresholds[self.D_INLIER_RATIOS.index(inlier_ratio)]

        if self.eval_mode:
            # Cache miss in eval mode
            raise FileNotFoundError(
                f"{self.CACHE_FILES['split_stats']} not found, run in training mode first."
            )

        # Calculate depth stats and create/add to cache
        split_hist, bin_edges = self.__compute_depth_histogram()
        depth_thresholds = self.__compute_depth_thresholds(split_hist, bin_edges)

        if self.cross_validation:
            depth_threshold_cache[fold_n] = depth_thresholds
        else:
            depth_threshold_cache = depth_thresholds

        with open(split_depth_cache, "w") as f:
            json.dump(depth_threshold_cache, f)

        return depth_thresholds[self.D_INLIER_RATIOS.index(inlier_ratio)]

    def __compute_class_weights(
        self,
        oversample: tuple[int, list[str]] = (0, []),
    ) -> torch.Tensor:
        print("Computing class weights...")
        # Init oversampling variables
        oversample_list = []
        oversample_labels = []
        for cls in oversample[1]:
            oversample_labels.append(self.CLASS_NAMES.index(cls))
        oversample_labels = torch.as_tensor(oversample_labels)

        # Counting loop
        class_counts = torch.zeros(self.NUM_CLASSES, dtype=torch.long)
        for idx in range(len(self)):
            image, _, _, _, semantics = self._load_sample(idx)
            if self.mask_black:
                where_black = image.sum(-1) == 0
                semantics[where_black] = self.ignore_index
            semantics = np.ascontiguousarray(semantics[np.newaxis, ...], dtype=np.int64)
            label = torch.as_tensor(semantics)

            # Mask out the ignore_label
            valid_labels = label[label != self.ignore_index]  # Flattenned

            # Count occurrences
            counts = torch.bincount(valid_labels, minlength=self.NUM_CLASSES)
            class_counts += counts

            # Oversampling
            if not (len(oversample_labels) and (counts[oversample_labels] > 0).any()):
                continue

            for _ in range(oversample[0]):
                class_counts += counts
                oversample_list.append(self.sample_list[idx])

        self.sample_list.extend(oversample_list)

        total = torch.sum(class_counts)
        class_weights = torch.tensor(
            # TODO CHECK
            [total / count for count in class_counts],
            dtype=torch.float32,
        )
        return class_weights

    def __compute_depth_histogram(self) -> tuple[np.ndarray, np.ndarray]:
        # Computes the combined depth histogram for all input areas/scenes
        area_depth_cache = self.dataset_path / self.CACHE_FILES["area_stats"]
        depth_hist_dict = None
        if area_depth_cache.exists():
            with open(area_depth_cache, "r") as f:
                depth_hist_dict = json.load(f)
        if not depth_hist_dict:
            raise FileNotFoundError(
                f"{self.CACHE_FILES['area_stats']} is corrupted, verify data integrity."
            )

        split_hist = np.array([])
        bin_edges = np.array([])
        for area in self.input_areas:
            area_path = self.dataset_path / area
            if (not area_path.exists()) or (not any(area_path.iterdir())):
                # Area/Scene folder is empty or doesn't exist
                continue
            hist = depth_hist_dict[area][0]
            bins = depth_hist_dict[area][1]
            if len(split_hist):
                split_hist += np.array(hist)
            else:
                split_hist = np.array(hist)
                bin_edges = np.array(bins)

        return split_hist, bin_edges

    def __compute_depth_thresholds(
        self, histogram: np.ndarray, bin_edges: np.ndarray
    ) -> list[float]:
        # Calculate the cumulative frequency
        cumulative_frequency = np.cumsum(histogram)
        total_frequency = cumulative_frequency[-1]

        bin_indices = []
        for ratio in self.D_INLIER_RATIOS:
            # Calculate the frequency of the inliers
            threshold = ratio * total_frequency
            # Find the bin where the cumulative frequency first exceeds the threshold
            bin_indices.append(
                np.where(cumulative_frequency >= threshold)[0][0]
            )  # Get the first index meeting the condition

        # Find the bin edge that corresponds to this bin
        return [float(bin_edges[bin_index]) for bin_index in bin_indices]


def load_cached(
    cache_file: Path, fold_n: int
) -> tuple[list[float] | None, dict[Any, list[float]]]:
    if cache_file.exists():
        with open(cache_file, "r") as f:
            cache_content = json.load(f)

        if isinstance(cache_content, list):
            return cache_content, {}
        elif isinstance(cache_content, dict):
            output = cache_content.get(str(fold_n), None)
            return output, cache_content

    return None, {}
    return None, {}
