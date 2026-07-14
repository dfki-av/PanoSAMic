"""
Standalone single-image demo for PanoSAMic.

Runs semantic segmentation on any panoramic image without downloading or
preprocessing a dataset. A hosted browser demo is also available at
https://huggingface.co/spaces/dfki-av/PanoSAMic-demo — this script is the
local/scriptable equivalent that writes results to disk instead.

Example:
    python scripts/demo.py --image my_panorama.jpg --output_dir demo_output/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from panosamic.datasets.stanford2d3ds import Stanford2d3dsDataset
from panosamic.evaluation.utils.config import parse_modalities
from panosamic.model import PanoSAMic
from panosamic.model.instance_semantic_fusion import refine_semantic_with_instances
from panosamic.palette import STANFORD2D3DS_COLORS, build_palette


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PanoSAMic on a single panoramic image, no dataset required."
    )
    parser.add_argument(
        "--image", type=Path, required=True, help="Path to the input panoramic image"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("demo_output"),
        help="Directory to write the segmentation outputs to",
    )
    parser.add_argument(
        "--checkpoint",
        default="dfki-av/PanoSAMic",
        help="Hub repo id or local path to a released PanoSAMic checkpoint",
    )
    parser.add_argument(
        "--subfolder",
        default="stanford2d3ds-vith-rgb-fold1",
        help=(
            "Subfolder within the Hub repo (default: the RGB-only Stanford2D3DS "
            "checkpoint, which needs no depth/normals input)"
        ),
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/config_stanford2d3ds_dv.json",
        help="Model architecture config matching the checkpoint",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="image",
        help="Comma-separated modalities the checkpoint expects (default: image-only)",
    )
    parser.add_argument(
        "--sam_weights_path",
        default=None,
        help="Path to SAM weights file or directory (auto-downloaded if omitted)",
    )
    parser.add_argument(
        "--vit_model",
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM encoder variant used by the checkpoint",
    )
    parser.add_argument(
        "--class_names",
        default=None,
        help=(
            "Comma-separated class names for the legend and color palette "
            "(default: Stanford2D3DS's 13 classes, matching the default checkpoint)"
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (default: cuda if available, else mps, else cpu)",
    )
    return parser.parse_args()


def _select_device(preferred: str | None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = _parse_args()
    device = _select_device(args.device)

    class_names = (
        args.class_names.split(",")
        if args.class_names
        else list(Stanford2d3dsDataset.CLASS_NAMES)
    )
    num_classes = len(class_names)

    print(
        f"Loading checkpoint {args.checkpoint!r} (subfolder={args.subfolder!r}) on {device}..."
    )
    model = PanoSAMic.from_pretrained_panosamic(
        args.checkpoint,
        sam_weights_path=args.sam_weights_path,
        vit_model=args.vit_model,
        config_path=args.config_path,
        num_classes=num_classes,
        subfolder=args.subfolder,
        modalities=parse_modalities(args.modalities),
    )
    model = model.to(device).eval()

    image = Image.open(args.image).convert("RGB")
    image_array = np.ascontiguousarray(
        np.array(image).transpose(2, 0, 1), dtype=np.float32
    )
    image_tensor = torch.as_tensor(image_array, device=device)

    print(f"Running inference on {args.image} ({image.width}x{image.height})...")
    with torch.no_grad():
        outputs = model([{"image": image_tensor}])

    output = outputs[0]
    sem_preds = output["sem_preds"]
    if output["instance_masks"]:
        sem_preds = refine_semantic_with_instances(
            sem_preds.squeeze(0), output["instance_masks"]
        ).unsqueeze(0)

    labels = torch.argmax(sem_preds, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

    palette = (
        STANFORD2D3DS_COLORS
        if class_names == list(Stanford2d3dsDataset.CLASS_NAMES)
        else build_palette(num_classes)
    )
    color_idx = np.clip(labels + 1, 0, palette.shape[0] - 1)
    color_image = palette[color_idx]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    seg_png_path = args.output_dir / f"{stem}_segmentation.png"
    seg_npy_path = args.output_dir / f"{stem}_segmentation.npy"
    legend_path = args.output_dir / f"{stem}_legend.json"

    Image.fromarray(color_image.astype(np.uint8), mode="RGB").save(seg_png_path)
    np.save(seg_npy_path, labels)
    with open(legend_path, "w") as fp:
        json.dump({str(i): name for i, name in enumerate(class_names)}, fp, indent=2)

    print(f"Saved segmentation image to {seg_png_path}")
    print(f"Saved class-id array to {seg_npy_path}")
    print(f"Saved class legend to {legend_path}")


if __name__ == "__main__":
    main()
