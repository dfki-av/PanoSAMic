"""
Evaluate PanoSAMic checkpoints across multiple folds and optionally save colorized predictions.

This script loads the trained model per fold, runs inference on the eval split,
computes mIoU/mAcc, and (if requested) saves prediction visualizations using
the dataset color map (or a provided colors.npy).

Example:
python scripts/eval_panosamic_multifold.py \
    --dataset_path /data/Datasets/Stanford2D3DS/Stanford2D3D_noXYZ/processed \
    --config_path config/config_stanford2d3ds_dv.json \
    --experiments_path experiments/ \
    --sam_weights_path sam_weights/sam_vit_h_4b8939.pth \
    --dataset stanford2d3ds \
    --vit_model vit_h \
    --modalities image \
    --folds 1 2 3 \
    --output runs/panosamic_eval_multifold.json \
    --save-dir visualizations/
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from panosamic.datasets import build_dataset
from panosamic.evaluation.metrics import compute_metrics, intersection_and_union_gpu
from panosamic.evaluation.utils.config import generate_configs
from panosamic.evaluation.utils.data import collate_as_lists
from panosamic.evaluation.utils.parser import create_parser
from panosamic.model import panosamic_builder
from panosamic.model.instance_semantic_fusion import (
    refine_semantic_with_instances,
)


def _edge_mask(height: int, width: int, side_frac: float, device: torch.device):
    """Boolean mask keeping left/right strips (side_frac of width each)."""
    if side_frac <= 0 or side_frac >= 0.5:
        return torch.ones((height, width), device=device, dtype=torch.bool)
    side_w = max(1, int(round(side_frac * width)))
    mask = torch.zeros((height, width), device=device, dtype=torch.bool)
    mask[:, :side_w] = True
    mask[:, width - side_w :] = True
    return mask


def _load_colors(path: Path, num_classes: int) -> np.ndarray | None:
    if not path.exists():
        return None
    colors = np.load(path)
    if colors.shape[0] == num_classes + 1 and (colors[0] == 0).all():
        return colors.astype(np.uint8)
    color_map = np.zeros((num_classes + 1, 3), dtype=np.uint8)
    color_map[1 : num_classes + 1] = colors[:num_classes]
    return color_map


def get_checkpoint_path(path: Path, id: str) -> Path | None:
    if id in str(path.name):
        return path
    all_exp = [dir for dir in path.glob(f"*{id}") if dir.is_dir()]
    all_exp.sort()
    if all_exp:
        return all_exp[-1]
    return None


def _save_pred_map(
    pred: torch.Tensor,
    sample_path: Path,
    save_root: Path,
    fold: int,
    color_map: np.ndarray,
    ignore_index: int,
):
    pred_np = pred.cpu().numpy().astype(np.int64)
    pred_np[pred_np == ignore_index] = -1
    color_idx = np.clip(pred_np + 1, 0, color_map.shape[0] - 1)
    color_img = color_map[color_idx]
    rel_path = sample_path.with_suffix(".png")
    out_path = save_root / f"fold_{fold}" / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color_img.astype(np.uint8), mode="RGB").save(out_path)


def _eval_fold(
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
    save_dir: Path | None,
    edge_side_fracs: list[float],
) -> dict[str, Any]:
    # Clone args to mutate fold
    fold_args = deepcopy(args)
    fold_args.fold = fold
    platform_config, training_config, model_config = generate_configs(args=fold_args)
    if args.refine:
        model_config.semantic_only = False

    dataset = build_dataset(
        dataset_path=platform_config.dataset_path,
        config=training_config,
        n_modalities=len(model_config.modalities),
        test_mode=True,
    )

    # Build model
    model = panosamic_builder(
        config=model_config,
        num_classes=dataset.NUM_CLASSES,
        freeze_encoder=True,
    )

    config_path = Path(platform_config.config_path)
    experiments_path = Path(platform_config.experiments_path)
    if experiments_path.name != training_config.dataset_name:
        experiments_path = experiments_path / training_config.dataset_name
    exp_id = (
        f"{config_path.stem}_F{training_config.fold}_V{model_config.vit_model[-1]}"
        f"_M{len(model_config.modalities)}"
    )
    ckpt_dir = get_checkpoint_path(path=experiments_path, id=exp_id)
    if not ckpt_dir:
        raise FileNotFoundError(
            f"No checkpoint dir found for id {exp_id} in {experiments_path}"
        )
    model_path = ckpt_dir / "model_best.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # Optionally load SAM weights for instance branch (before trained checkpoint)
    if args.refine and platform_config.sam_weights_path:
        sam_checkpoint = torch.load(
            platform_config.sam_weights_path, map_location="cpu"
        )
        model.load_state_dict(sam_checkpoint, strict=False)

    # Load trained checkpoint (lenient to keep SAM-init params)
    checkpoint = torch.load(model_path, map_location="cpu")
    model_weights = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(model_weights, strict=False)
    epoch = checkpoint.get("epoch", -1)

    model = model.to(device)
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size if training_config.batch_size > 0 else 1,
        shuffle=False,
        num_workers=platform_config.workers_per_gpu,
        collate_fn=collate_as_lists,
        drop_last=False,
    )

    color_map = None
    if save_dir:
        # Prefer dataset-specific colors if present
        ds_colors = dataset.dataset_path / "assets" / "colors.npy"
        color_map = _load_colors(
            ds_colors if ds_colors.exists() else args.colors_path, dataset.NUM_CLASSES
        )

    area_intersection = torch.zeros(dataset.NUM_CLASSES, device=device)
    area_union = torch.zeros(dataset.NUM_CLASSES, device=device)
    area_target = torch.zeros(dataset.NUM_CLASSES, device=device)
    frame_metrics: list[dict[str, Any]] = []
    edge_accumulators: dict[float, dict[str, torch.Tensor]] = {
        frac: {
            "inter": torch.zeros(dataset.NUM_CLASSES, device=device),
            "union": torch.zeros(dataset.NUM_CLASSES, device=device),
            "target": torch.zeros(dataset.NUM_CLASSES, device=device),
        }
        for frac in edge_side_fracs
    }

    pbar = tqdm(
        total=len(dataloader), desc=f"Fold {fold} (Ep {epoch})", dynamic_ncols=True
    )
    with torch.no_grad():
        for batch_idx, (vinputs, vlabels) in enumerate(dataloader):
            # Move to device
            for i in range(len(vinputs)):
                for k, v in vinputs[i].items():
                    if isinstance(v, torch.Tensor):
                        vinputs[i][k] = v.to(device)
            for i in range(len(vlabels)):
                for k, v in vlabels[i].items():
                    if isinstance(v, torch.Tensor):
                        vlabels[i][k] = v.to(device)

            voutputs = model(vinputs)

            if args.refine:
                for output in voutputs:
                    semantic = output.get("sem_preds")
                    instances = output.get("instance_masks", [])
                    if semantic is None or not len(instances):
                        continue
                    has_batch = semantic.dim() == 4
                    semantic_nb = semantic.squeeze(0) if has_batch else semantic
                    refined = refine_semantic_with_instances(semantic_nb, instances)
                    if has_batch:
                        refined = refined.unsqueeze(0)
                    output["sem_preds"] = refined

            metrics = compute_metrics(
                pred_list=voutputs,
                label_list=vlabels,
                num_classes=dataset.NUM_CLASSES,
                ignore_index=dataset.ignore_index,
                device=device,
            )
            area_intersection += metrics[0]
            area_union += metrics[1]
            area_target += metrics[2]

            effective_bs = dataloader.batch_size or len(voutputs) or 1

            for i, output in enumerate(voutputs):
                sem_pred = output["sem_preds"]
                pred_labels = torch.argmax(sem_pred, dim=1).squeeze(0)
                gt = vlabels[i]["semantics"].squeeze(0)
                sample_idx = batch_idx * effective_bs + i
                if (
                    save_dir
                    and color_map is not None
                    and sample_idx < len(dataset.sample_list)
                ):
                    rel = dataset.sample_list[sample_idx].relative_to(
                        dataset.dataset_path
                    )
                    _save_pred_map(
                        pred=pred_labels,
                        sample_path=rel,
                        save_root=save_dir,
                        fold=fold,
                        color_map=color_map,
                        ignore_index=dataset.ignore_index,
                    )

                for frac, acc in edge_accumulators.items():
                    mask = _edge_mask(
                        height=pred_labels.shape[0],
                        width=pred_labels.shape[1],
                        side_frac=frac,
                        device=device,
                    )
                    masked_pred = pred_labels.clone()
                    masked_gt = gt.clone()
                    masked_pred[~mask] = dataset.ignore_index
                    masked_gt[~mask] = dataset.ignore_index
                    inter_edge, uni_edge, tgt_edge = intersection_and_union_gpu(
                        output=masked_pred,
                        target=masked_gt,
                        num_classes=dataset.NUM_CLASSES,
                        ignore_index=dataset.ignore_index,
                    )
                    acc["inter"] += inter_edge
                    acc["union"] += uni_edge
                    acc["target"] += tgt_edge

            # Frame-level IoU (per batch item)
            for i, output in enumerate(voutputs):
                sem_pred = output["sem_preds"]
                pred_labels = torch.argmax(sem_pred, dim=1).squeeze(0)
                gt = vlabels[i]["semantics"].squeeze(0)
                inter, uni, _ = intersection_and_union_gpu(
                    output=pred_labels,
                    target=gt,
                    num_classes=dataset.NUM_CLASSES,
                    ignore_index=dataset.ignore_index,
                )
                frame_iou = inter / uni
                sample_idx = batch_idx * effective_bs + i
                if sample_idx < len(dataset.sample_list):
                    sample_id = (
                        dataset.sample_list[sample_idx]
                        .relative_to(dataset.dataset_path)
                        .as_posix()
                    )
                else:
                    sample_id = f"batch{batch_idx}_idx{i}"
                frame_metrics.append(
                    {"sample": sample_id, "miou": torch.nanmean(frame_iou).item()}
                )

            pbar.update()
    pbar.close()

    iou_per_class = area_intersection / area_union
    acc_per_class = area_intersection / area_target
    miou = torch.nanmean(iou_per_class).item()
    macc = torch.nanmean(acc_per_class).item()

    edge_results: dict[str, Any] = {}
    for frac, acc in edge_accumulators.items():
        edge_iou = acc["inter"] / acc["union"]
        edge_acc = acc["inter"] / acc["target"]
        key = f"{int(round(frac * 100))}pct_per_side"
        edge_results[key] = {
            "miou": torch.nanmean(edge_iou).item(),
            "macc": torch.nanmean(edge_acc).item(),
            "iou_per_class": {
                name: torch.nan_to_num(val, nan=float("nan")).item()
                for name, val in zip(dataset.CLASS_NAMES, edge_iou)
            },
        }

    return {
        "fold": fold,
        "epoch": epoch,
        "miou": miou,
        "macc": macc,
        "iou_per_class": {
            name: torch.nan_to_num(val, nan=float("nan")).item()
            for name, val in zip(dataset.CLASS_NAMES, iou_per_class)
        },
        "frame_metrics": frame_metrics,
        "edge_metrics": edge_results,
    }


def main() -> None:
    parser = create_parser()
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Which folds to evaluate (default: use --fold; or 1 2 3 if not set).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/panosamic_eval_multifold.json"),
        help="Where to store aggregated metrics.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Optional directory to save colorized predictions.",
    )
    parser.add_argument(
        "--colors-path",
        type=Path,
        default=Path("colors.npy"),
        help="Path to colors.npy used for visualization.",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Apply instance-guided refinement if instance masks are available.",
    )
    parser.add_argument(
        "--use-baseline-decoder",
        action="store_true",
        help="Force BaselineDecoder (sets basic_fusion to 'concat').",
    )
    parser.add_argument(
        "--edge-side-fracs",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Evaluate edge-only IoU on left/right strips (fractional width per side). "
            "Example: --edge-side-fracs 0.25 0.15 0.05 keeps 25%, 15%, 5% of width on each side."
        ),
    )
    args = parser.parse_args()

    if args.folds is None:
        folds = [args.fold] if args.fold else [1, 2, 3]
    else:
        folds = args.folds
    edge_side_fracs = []
    if args.edge_side_fracs:
        edge_side_fracs = sorted(
            {f for f in args.edge_side_fracs if 0 < f < 0.5}, reverse=True
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, Any] = {
        "device": str(device),
        "folds": {},
        "edge_side_fracs": edge_side_fracs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        fold_result = _eval_fold(
            fold=fold,
            args=args,
            device=device,
            save_dir=args.save_dir,
            edge_side_fracs=edge_side_fracs,
        )
        results["folds"][fold] = fold_result

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
