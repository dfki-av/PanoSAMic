"""
Export a PanoSAMic training checkpoint to a Hub-ready trainable-only safetensors file.

Strips the frozen SAM backbone (image_encoder, prompt_encoder, mask_decoder) and
writes only the trainable parameters (feature_fuser, semantic_decoder).

Usage
-----
    uv run python scripts/export_checkpoint_for_hub.py \\
        --input  ./experiments/stanford2d3ds/.../model_best.pth \\
        --output ./exports/stanford2d3ds-vith-rgbdn-fold1/model.safetensors

    # Verify the export without SAM weights or datasets:
    uv run python scripts/export_checkpoint_for_hub.py --self-test \\
        --input  ./experiments/stanford2d3ds/.../model_best.pth
"""

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file as safetensors_save

# Keys belonging to the frozen SAM backbone — strip these before publishing.
_FROZEN_PREFIXES = ("image_encoder.", "prompt_encoder.", "mask_decoder.")


def strip_backbone(full_state: dict) -> dict[str, torch.Tensor]:
    """Remove frozen SAM backbone keys from a full model state_dict."""
    return {k: v for k, v in full_state.items() if not k.startswith(_FROZEN_PREFIXES)}


def export(input_path: Path, output_path: Path) -> dict[str, torch.Tensor]:
    """Load a training checkpoint, strip the backbone, save as safetensors."""
    raw = torch.load(input_path, map_location="cpu", weights_only=True)
    full_state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw

    trainable = strip_backbone(full_state)

    backbone_keys = [k for k in full_state if k.startswith(_FROZEN_PREFIXES)]
    print(f"Stripped {len(backbone_keys)} frozen backbone keys.")
    print(f"Retaining {len(trainable)} trainable keys.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    safetensors_save(trainable, str(output_path))
    size_mb = output_path.stat().st_size / 1e6
    print(f"Saved to {output_path}  ({size_mb:.1f} MB)")

    return trainable


def self_test(input_path: Path) -> None:
    """Export and reload; assert backbone-free and all trainable keys round-trip.

    Does NOT download SAM weights or build the full model — verifies only the
    trainable key export/import cycle using safetensors directly.
    """
    import tempfile

    from safetensors.torch import load_file as safetensors_load

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "model.safetensors"
        trainable = export(input_path, out)

        leaked = [k for k in trainable if k.startswith(_FROZEN_PREFIXES)]
        if leaked:
            raise RuntimeError(f"Backbone keys leaked into export: {leaked}")
        print("✓ Export is backbone-free.")

        reloaded = safetensors_load(str(out), device="cpu")
        missing = set(trainable) - set(reloaded)
        if missing:
            raise RuntimeError(f"Keys missing after reload: {missing}")
        for k in trainable:
            if trainable[k].shape != reloaded[k].shape:
                raise RuntimeError(f"Shape mismatch for {k}")
        print(f"✓ All {len(trainable)} trainable keys reloaded with correct shapes.")
        print("Self-test passed.")


def push_to_hub(
    safetensors_path: Path,
    repo_id: str,
    *,
    path_in_repo: str = "model.safetensors",
    commit_message: str = "Upload PanoSAMic trainable checkpoint",
) -> None:
    """Upload an exported safetensors file to a Hugging Face Hub repo.

    The repo must already exist.  Create it first with:
        huggingface-cli repo create <repo_id> --type model

    Use *path_in_repo* to place the file in a subfolder, e.g.:
        path_in_repo="stanford2d3ds-vith-rgbdn/model.safetensors"
    """
    from huggingface_hub import HfApi

    api = HfApi()
    url = api.upload_file(
        path_or_fileobj=str(safetensors_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    print(f"Uploaded to: {url}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Path to model_best.pth from training"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output safetensors path (default: <input_dir>/model.safetensors)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Export + reload and assert correctness (no datasets, no SAM download needed)",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        print(f"Error: {input_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    output_path: Path = args.output or (input_path.parent / "model.safetensors")

    if args.self_test:
        self_test(input_path)
    else:
        export(input_path, output_path)


if __name__ == "__main__":
    main()
