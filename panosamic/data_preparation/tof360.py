"""
Preprocessing Script for the ToF-360 dataset

Author: Mahdi Chamseddine
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from shutil import copytree

import numpy as np
from PIL import Image
from tqdm import tqdm

from panosamic.datasets.base import BaseDataset
from panosamic.palette import STANFORD2D3DS_COLORS

ALL_SCENES = ("Hospital", "Office_Room_1", "Office_Room_2", "Parking_Lot")


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset_root",
        required=True,
        help="Path to the ToF-360 dataset",
    )
    parser.add_argument(
        "-o",
        "--output_root",
        required=True,
        help="Path to the directory where the output will be saved",
    )
    parser.add_argument(
        "-w",
        "--target_width",
        type=int,
        default=1024,
        help=(
            "Target output width in pixels, height scaled to match aspect ratio "
            "(default: 1024, matching SAM's native img_size -- ToF-360's raw "
            "images are 5792px wide, far larger than SAM's encoder resolution, "
            "so downscaling keeps preprocessing/eval memory and disk usage "
            "reasonable). Pass 0 to keep the original size."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing preprocessed data",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )

    return parser


def read_sample_names(dataset_dir: Path, scene: str) -> list[str]:
    suffix = f"_{scene}_equi_rgb.png"
    return sorted(
        f.name[: -len(suffix)]
        for f in (dataset_dir / scene / "RGB").iterdir()
        if f.name.endswith(suffix)
    )


def process_sample(
    scene_dir: Path,
    scene_output_dir: Path,
    scene: str,
    sample_name: str,
    target_width: int,
    save_depth_stats: bool,
) -> list[np.ndarray]:
    stem = f"{sample_name}_{scene}_equi"
    input_sizes = []

    path = scene_dir / "RGB" / f"{stem}_rgb.png"
    image_rgb = Image.open(path.absolute(), "r").convert("RGB")
    input_sizes.append(image_rgb.size)

    path = scene_dir / "depth" / f"{stem}_depth.png"
    depth = Image.open(path.absolute(), "r")
    input_sizes.append(depth.size)

    path = scene_dir / "normal" / f"{stem}_normal.png"
    normals = Image.open(path.absolute(), "r").convert("RGB")
    input_sizes.append(normals.size)

    path = scene_dir / "semantics" / f"{stem}_semantic.npy"
    semantics_arr = np.load(path.absolute())
    # np.ndarray.shape is (H, W); PIL .size is (W, H) -- match the latter for the
    # dimension-consistency check below.
    input_sizes.append((semantics_arr.shape[1], semantics_arr.shape[0]))

    input_sizes = set(input_sizes)
    if len(input_sizes) != 1:
        raise ValueError(
            f"Dimensions of input images do not match in sample: {scene}/{sample_name}"
        )

    depth_mask = Image.fromarray(  # 0 indicates no reading
        np.array(depth, dtype=np.uint16) != 0
    )

    # semantics.npy carries ToF-360's own raw, dataset-wide category ids and is
    # copied through unchanged here -- the Stanford-taxonomy collapse happens at
    # load time in ToF360Dataset._map_semantic_labels, mirroring how
    # Matterport3dDataset defers its own taxonomy mapping to the loader instead
    # of baking it into preprocessing.
    semantics = Image.fromarray(semantics_arr)

    orig_w, orig_h = input_sizes.pop()
    if target_width and target_width < orig_w:
        new_size = (target_width, round(orig_h * target_width / orig_w))
        image_rgb = image_rgb.resize(new_size, resample=Image.Resampling.LANCZOS)
        depth = depth.resize(new_size, resample=Image.Resampling.LANCZOS)
        depth_mask = depth_mask.resize(new_size, resample=Image.Resampling.NEAREST)
        normals = normals.resize(new_size, resample=Image.Resampling.NEAREST)
        semantics = semantics.resize(new_size, resample=Image.Resampling.NEAREST)

    depth_m = np.array(depth, int) / 512  # depth is in 1/512m units, 0 = no reading

    sample_output_dir = scene_output_dir / sample_name
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    # webp is used for its compression properties of lossless images
    # webp can't handle 16bit channels so png is used instead for depth
    ext = "webp"
    image_rgb.save(sample_output_dir / f"rgb.{ext}", format=ext, lossless=True)
    depth.save(sample_output_dir / "depth.png", format="png", lossless=True)
    depth_mask.save(sample_output_dir / f"depth_mask.{ext}", format=ext, lossless=True)
    normals.save(sample_output_dir / f"normals.{ext}", format=ext, lossless=True)
    semantics.save(sample_output_dir / "semantics.png", format="png", lossless=True)

    if save_depth_stats:
        masked_depth = np.ma.array(depth_m, mask=np.array(depth_mask) == 0)
        masked_depth = masked_depth.compressed()
        max_depth = round(np.iinfo(np.uint16).max / 512)
        depth_hist, bin_edges = np.histogram(
            masked_depth, bins=max_depth * 10, range=(0, max_depth)
        )

        return [depth_hist, np.round(bin_edges, decimals=1)]

    return []


def plot_histogram(
    plt,
    histogram: np.ndarray,
    edge_bins: np.ndarray,
    desc: str,
) -> None:
    # Plot the histogram
    plt.cla()
    plt.clf()
    plt.bar(
        edge_bins,
        histogram,
        width=0.1,
        align="edge",
    )
    plt.xlabel("Depth (m)")
    plt.ylabel("Frequency")
    plt.title(desc)
    plt.show()


def main():
    print("Preprocessing ToF-360 dataset")

    parser = create_parser()
    args = parser.parse_args()

    if args.debug:
        print(f"-- Passed arguments: {args}")

        try:
            import matplotlib.pyplot as plt  # type:ignore
        except ImportError:
            import warnings

            warnings.warn(
                "Matplotlib must be installed for plotting histograms!", stacklevel=2
            )
            print("Proceeding without plotting...")
            args.debug = False

    dataset_dir = Path(args.dataset_root)

    # Verify data validity
    dir_contents = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
    scene_list = [d for d in dir_contents if d in ALL_SCENES]
    scene_list.sort()
    if not scene_list:
        raise FileNotFoundError(f"No ToF-360 scenes were found in {dataset_dir}")

    output_dir = Path(args.output_root)
    if not output_dir.is_dir():
        print(f"{output_dir} doesn't exist, creating...")
        output_dir.mkdir(parents=True, exist_ok=True)

    # A full raw download includes a top-level 'assets' folder (preprocessing/eval
    # scripts, figures -- no label data, unlike Stanford2D3DS/Matterport3D). It's
    # not required by anything downstream, so copy it through if present (e.g. a
    # full `git clone`/`snapshot_download`) but don't require it -- selectively
    # downloading only the scene subfolders needed is a valid, common workflow.
    if "assets" in dir_contents:
        copytree(dataset_dir / "assets", output_dir / "assets", dirs_exist_ok=True)
    (output_dir / "assets").mkdir(parents=True, exist_ok=True)

    # ToF360Dataset reuses Stanford2D3DS's 13-class taxonomy (see its docstring),
    # so this recreates Stanford2D3DS's colors.npy here rather than requiring it
    # to be copied in manually, so PanoSAMicEvaluator's colorized-output path
    # (which expects dataset_path/assets/colors.npy) works out of the box.
    colors_path = output_dir / "assets" / "colors.npy"
    if not colors_path.exists():
        np.save(colors_path, STANFORD2D3DS_COLORS)

    cache_file = output_dir / "cache_samples_file_names.json"
    scene_dict = {}
    # If the cache file exists and not to be rewritten, then load it
    if args.overwrite or not cache_file.exists():
        print("-- Reading file names: scanning files")
        for scene in scene_list:
            scene_dict[scene] = read_sample_names(dataset_dir, scene)

        with open(cache_file, "w") as fp:
            json.dump(scene_dict, fp)
    else:
        print("-- Reading file names: loading cached names")
        with open(cache_file) as fp:
            scene_dict = json.load(fp)

    depth_hist_dict = {}
    cache_file = output_dir / "cache_area_depth_statistics.json"
    # If the cache file exists and not to be rewritten, then don't do the computations
    print("-- Processing samples")
    save_depth_stats = args.overwrite or not cache_file.exists()
    for scene, sample_list in scene_dict.items():
        depth_hist_dict[scene] = []
        scene_output_dir = output_dir / scene
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        for sample_name in tqdm(
            sample_list,
            desc=f"  -- Processing {scene}",
            ncols=100,
            unit=" samples",
        ):
            sample_depth_hist = process_sample(
                dataset_dir / scene,
                scene_output_dir,
                scene,
                sample_name,
                args.target_width,
                save_depth_stats,
            )
            if save_depth_stats:
                try:
                    depth_hist_dict[scene][0] += sample_depth_hist[0]
                    # Bins are the same no need to save them
                except IndexError:
                    depth_hist_dict[scene] = sample_depth_hist

        if args.debug and save_depth_stats:
            plot_histogram(
                plt,
                depth_hist_dict[scene][0],
                depth_hist_dict[scene][1][:-1],
                f"Histogram of depths in {scene}",
            )

    if save_depth_stats:
        # Combine scenes into one histogram without aliasing the per-scene arrays
        # (they're still needed below for the per-scene cache file). Scenes with
        # no samples leave an empty [] placeholder in depth_hist_dict -- skip those.
        combined_hist: np.ndarray | None = None
        combined_bin_edges: np.ndarray | None = None
        for value in depth_hist_dict.values():
            if not value:
                continue
            hist, bin_edges = value
            if combined_hist is None:
                combined_hist = np.array(hist, dtype=np.int64)
                combined_bin_edges = np.array(bin_edges)
            else:
                combined_hist = combined_hist + np.array(hist, dtype=np.int64)

        if combined_hist is None or combined_bin_edges is None:
            raise ValueError(
                "No samples were processed; cannot compute depth statistics."
            )

        if args.debug:
            plot_histogram(
                plt,
                combined_hist,
                combined_bin_edges[:-1],
                "Histogram of depths in all scenes",
            )

        print("-- Saving depth statistics")
        cache_dict = {}
        for key, value in depth_hist_dict.items():
            cache_dict[key] = [nparr.tolist() for nparr in value]

        with open(cache_file, "w") as fp:
            json.dump(cache_dict, fp)

        # ToF-360 is eval-only: BaseDataset._get_depth_threshold() raises
        # FileNotFoundError in eval mode if cache_splits_depth_statistics.json is
        # missing, since it's normally populated by a training-mode pass -- which
        # ToF-360 never has (ToF360Dataset's fold 1 uses all scenes for both train
        # and eval). Compute and write it here instead of leaving it to fail at
        # evaluation time.
        print("-- Computing and saving eval-mode depth thresholds")
        cumulative_frequency = np.cumsum(combined_hist)
        total_frequency = cumulative_frequency[-1]
        thresholds = []
        for ratio in BaseDataset.D_INLIER_RATIOS:
            threshold = ratio * total_frequency
            bin_index = np.where(cumulative_frequency >= threshold)[0][0]
            thresholds.append(float(combined_bin_edges[bin_index]))

        with open(output_dir / "cache_splits_depth_statistics.json", "w") as fp:
            json.dump(thresholds, fp)


if __name__ == "__main__":
    main()
