"""
Unified SAM3 evaluation for panosamic datasets (Stanford2D3DS, Matterport3D).

Features:
  - Runs SAM3 text prompts per class and fuses masks.
  - Optional clutter handling (Stanford adds a clutter channel).
  - Optional coverage thresholding and majority smoothing to reduce speckle.
  - Optional visualization using dataset colors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from panosamic.datasets.matterport3d import Matterport3dDataset
from panosamic.datasets.stanford2d3ds import Stanford2d3dsDataset
from panosamic.evaluation.metrics import intersection_and_union_gpu


def _ensure_sam3_import(repo_hint: Path | None) -> None:
    try:
        import sam3  # noqa: F401 # type: ignore

        return
    except ImportError:
        pass

    candidates: list[Path] = []
    if repo_hint:
        candidates.append(repo_hint)
    default_repo = Path(__file__).resolve().parents[2] / "sam3"
    candidates.append(default_repo)

    for candidate in candidates:
        if candidate and candidate.exists():
            sys.path.append(str(candidate))
            break

    import sam3  # noqa: F401 # type: ignore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 on panosamic datasets (Stanford2D3DS, Matterport3D)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["stanford2d3ds", "matterport3d"],
        required=True,
        help="Dataset name.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the processed dataset root.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to sam3.pt checkpoint.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Which folds to evaluate (default: dataset-specific default).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for SAM3 (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold used by SAM3 to filter mask predictions.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.05,
        help="Minimum per-pixel class score; pixels below are set to clutter/ignore.",
    )
    parser.add_argument(
        "--smooth-kernel",
        type=int,
        default=0,
        help="Apply majority smoothing with an odd kernel size (e.g., 3 or 5). Use 0 to disable.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1008,
        help="Resolution used by Sam3Processor for resizing.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/sam3_eval_panosamic.json"),
        help="Where to write aggregated metrics (JSON).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Optional directory to save colorized prediction maps.",
    )
    parser.add_argument(
        "--colors-path",
        type=Path,
        default=Path("colors.npy"),
        help="Fallback path to colors.npy if dataset assets do not contain one.",
    )
    parser.add_argument(
        "--sam3-repo",
        type=Path,
        help="Optional path to a local SAM3 clone if it is not installed.",
    )
    return parser.parse_args()


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.numpy().transpose(1, 2, 0)
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _majority_filter(labels: torch.Tensor, kernel: int) -> torch.Tensor:
    """Apply a simple majority vote in a sliding window to smooth labels."""
    if kernel <= 1:
        return labels
    pad = kernel // 2
    unfolded = F.unfold(
        labels.unsqueeze(0).unsqueeze(0).float(), kernel_size=kernel, padding=pad
    ).squeeze(0)
    mode_vals = torch.mode(unfolded.long(), dim=0).values
    h, w = labels.shape
    return mode_vals.view(h, w)


def _predict_semantics_for_image(
    processor,
    image: Image.Image,
    class_names: Sequence[str],
    clutter_idx: int | None,
    coverage_threshold: float,
    smooth_kernel: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run SAM3 once per class and return a (C, H, W) tensor of class scores.

    If clutter_idx is provided, clutter channel is filled where no/low coverage.
    Otherwise, coverage thresholding is applied to labels as ignore later.
    """
    base_state: dict = processor.set_image(image, state={})
    num_classes = len(class_names)
    height, width = image.height, image.width
    sem_pred = torch.zeros((num_classes, height, width), device=processor.device)

    for idx, cls in enumerate(class_names):
        if clutter_idx is not None and idx == clutter_idx:
            continue

        prompt_state = processor.set_text_prompt(prompt=cls, state=base_state)
        masks = prompt_state.get("masks_logits", None)
        scores = prompt_state.get("scores", None)
        processor.reset_all_prompts(base_state)

        if masks is None or scores is None or masks.numel() == 0:
            continue

        masks = masks.squeeze(1)  # (N, H, W)
        scores = scores.view(-1, 1, 1)
        sem_pred[idx] = (masks * scores).amax(dim=0)

    coverage = sem_pred.max(dim=0).values
    if clutter_idx is not None:
        sem_pred[clutter_idx] = (coverage <= 0).float()
        if coverage_threshold > 0:
            sem_pred[clutter_idx] = torch.where(
                coverage < coverage_threshold,
                torch.ones_like(sem_pred[clutter_idx]),
                sem_pred[clutter_idx],
            )

    if smooth_kernel and smooth_kernel % 2 == 1 and smooth_kernel > 1:
        labels = torch.argmax(sem_pred, dim=0)
        labels = _majority_filter(labels, kernel=smooth_kernel)
        sem_pred = F.one_hot(labels, num_classes=num_classes).permute(2, 0, 1).float()
    return sem_pred, coverage


def _load_colors(path: Path, fallback_classes: int) -> np.ndarray | None:
    if not path.exists():
        return None
    colors = np.load(path)
    if colors.shape[0] == fallback_classes + 1 and (colors[0] == 0).all():
        return colors.astype(np.uint8)
    color_map = np.zeros((fallback_classes + 1, 3), dtype=np.uint8)
    color_map[1 : fallback_classes + 1] = colors[:fallback_classes]
    return color_map


def _evaluate_fold(
    processor,
    dataset,
    class_names: list[str],
    clutter_idx: int | None,
    fold_n: int,
    coverage_threshold: float,
    smooth_kernel: int,
    save_dir: Path | None,
    color_map: np.ndarray | None,
) -> dict:
    device = processor.device
    num_classes = dataset.NUM_CLASSES
    area_intersection = torch.zeros(num_classes, device=device)
    area_union = torch.zeros(num_classes, device=device)
    area_target = torch.zeros(num_classes, device=device)
    frame_metrics: list[dict] = []

    for idx in tqdm(range(len(dataset)), desc=f"Fold {fold_n}"):
        inputs, labels = dataset[idx]
        rgb = _tensor_to_pil(inputs["image"])
        gt = labels["semantics"].to(device)

        sem_pred, coverage = _predict_semantics_for_image(
            processor=processor,
            image=rgb,
            class_names=class_names,
            clutter_idx=clutter_idx,
            coverage_threshold=coverage_threshold,
            smooth_kernel=smooth_kernel,
        )

        pred_labels = torch.argmax(sem_pred, dim=0)
        if clutter_idx is None:
            pred_labels = pred_labels.clone()
            coverage = coverage.to(device)
            if coverage_threshold > 0:
                pred_labels[coverage < coverage_threshold] = dataset.ignore_index

        if save_dir and color_map is not None:
            sample_rel = dataset.sample_list[idx].relative_to(dataset.dataset_path)
            save_path = save_dir / f"fold_{fold_n}" / sample_rel.with_suffix(".png")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            label_np = pred_labels.cpu().numpy().astype(np.int64)
            color_idx = np.clip(label_np + 1, 0, color_map.shape[0] - 1)
            color_img = color_map[color_idx]
            Image.fromarray(color_img.astype(np.uint8), mode="RGB").save(save_path)

        intersection, union, target = intersection_and_union_gpu(
            output=pred_labels,
            target=gt.squeeze(0),
            num_classes=num_classes,
            ignore_index=dataset.ignore_index,
        )

        area_intersection += intersection
        area_union += union
        area_target += target

        frame_iou = intersection / union
        frame_miou = torch.nanmean(frame_iou).item()
        sample_id = (
            dataset.sample_list[idx].relative_to(dataset.dataset_path).as_posix()
        )
        frame_metrics.append({"sample": sample_id, "miou": frame_miou})

    iou_per_class = area_intersection / area_union
    acc_per_class = area_intersection / area_target

    return {
        "miou": torch.nanmean(iou_per_class).item(),
        "macc": torch.nanmean(acc_per_class).item(),
        "iou_per_class": {
            name: torch.nan_to_num(val, nan=float("nan")).item()
            for name, val in zip(class_names, iou_per_class)
        },
        "frame_metrics": frame_metrics,
    }


def _build_dataset(dataset_name: str, dataset_path: Path, fold: int, eval_mode=True):
    if dataset_name == "stanford2d3ds":
        return Stanford2d3dsDataset(
            dataset_path=dataset_path,
            fold_n=fold,
            eval_mode=eval_mode,
            mask_black=True,
            semantic_only=True,
        )
    elif dataset_name == "matterport3d":
        return Matterport3dDataset(
            dataset_path=dataset_path,
            fold_n=fold,
            eval_mode=eval_mode,
            semantic_only=True,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def main() -> None:
    args = _parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    _ensure_sam3_import(args.sam3_repo)
    from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore
    from sam3.model_builder import build_sam3_image_model  # type: ignore

    model = build_sam3_image_model(
        device=device,
        checkpoint_path=str(args.checkpoint),
        eval_mode=True,
    )
    processor = Sam3Processor(
        model,
        resolution=args.resolution,
        device=device,
        confidence_threshold=args.confidence,
    )

    # Determine folds per dataset
    if args.folds is None:
        folds = [1, 2, 3] if args.dataset == "stanford2d3ds" else [1]
    else:
        folds = args.folds

    results = {"device": device, "dataset": args.dataset, "folds": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        dataset = _build_dataset(
            dataset_name=args.dataset,
            dataset_path=args.dataset_path,
            fold=fold,
            eval_mode=True,
        )
        class_names = list(dataset.CLASS_NAMES)
        clutter_idx = None
        if "clutter" in class_names:
            # Stanford2D3DS uses "objects" as the catch-all class
            clutter_idx = class_names.index("clutter")
        elif "objects" in class_names:
            # Matterport3D uses "objects" as the catch-all class
            clutter_idx = class_names.index("objects")

        color_map = None
        if args.save_dir:
            ds_colors = dataset.dataset_path / "assets" / "colors.npy"
            color_map = _load_colors(
                ds_colors if ds_colors.exists() else args.colors_path,
                fallback_classes=dataset.NUM_CLASSES,
            )

        fold_result = _evaluate_fold(
            processor=processor,
            dataset=dataset,
            class_names=class_names,
            clutter_idx=clutter_idx,
            fold_n=fold,
            coverage_threshold=args.coverage_threshold,
            smooth_kernel=args.smooth_kernel,
            save_dir=args.save_dir,
            color_map=color_map,
        )
        results["folds"][fold] = fold_result

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
