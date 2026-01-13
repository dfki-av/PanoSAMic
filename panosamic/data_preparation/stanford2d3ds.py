"""
Preprocessing Script for the Stanford-2D-3D Semantic dataset

Author: Mahdi Chamseddine
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from shutil import copytree

import numpy as np
from PIL import Image
from tqdm import tqdm


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset_root",
        required=True,
        help="Path to the Stanford 2D3D Semantics dataset",
    )
    parser.add_argument(
        "-o",
        "--output_root",
        required=True,
        help="Path to the directory where the output will be saved",
    )
    parser.add_argument(
        "-r",
        "--resize_image",
        nargs="?",
        const=0.25,
        default=1,
        type=float,
        help="Resize scale of the images [0.1-1]. If provided without value 0.25 is chosen",
    )
    # parser.add_argument(
    #     "--make_dirs",
    #     action="store_true",
    #     help="Create missing directories for output, if not passed an error will be thrown instead",
    # )
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


def read_sample_names(dir: Path, ext: str = ".png") -> list[str]:
    # Checks if the extension matches
    # Splits the string by underscores
    # Merges the strings without the past after the last underscore
    # Returns list of file names
    return ["_".join(f.stem.split("_")[0:-1]) for f in dir.iterdir() if f.suffix == ext]


def process_sample(
    area_pano_dir: Path,
    area_output_dir: Path,
    sample_name: str,
    scale: float,
    save_depth_stats: bool,
) -> list[np.ndarray]:
    input_sizes = []
    path = area_pano_dir / "rgb" / (sample_name + "_rgb.png")
    image_rgb = Image.open(path.absolute(), "r").convert("RGB")
    input_sizes.append(image_rgb.size)

    path = area_pano_dir / "depth" / (sample_name + "_depth.png")
    depth = Image.open(path.absolute(), "r")
    input_sizes.append(depth.size)

    path = area_pano_dir / "normal" / (sample_name + "_normals.png")
    normals = Image.open(path.absolute(), "r").convert("RGB")
    input_sizes.append(normals.size)

    path = area_pano_dir / "semantic" / (sample_name + "_semantic.png")
    instances = Image.open(path.absolute(), "r").convert("RGB")
    input_sizes.append(instances.size)

    input_sizes = set(input_sizes)
    assert len(input_sizes) == 1, (
        "Dimensions of input images do not match in sample: "
        + f"{area_pano_dir.name}/{sample_name}"
    )

    depth_mask = Image.fromarray(  # (256 * 256 - 1)  # Invalid or max depth
        np.array(depth, dtype=np.uint16) != 65535
    )

    if scale < 1:
        new_size = (np.array(input_sizes.pop()) * scale).astype(int)
        image_rgb = image_rgb.resize(new_size, resample=Image.Resampling.LANCZOS)
        depth = depth.resize(new_size, resample=Image.Resampling.LANCZOS)
        depth_mask = depth_mask.resize(new_size, resample=Image.Resampling.NEAREST)
        normals = normals.resize(new_size, resample=Image.Resampling.NEAREST)
        instances = instances.resize(new_size, resample=Image.Resampling.NEAREST)

    depth_m = (  # approximately: (256 * 256 - 1) * 128  # 16 bit -> 128m
        np.array(depth, int) * 128 / (256 * 256 - 1)
    )

    sample_output_dir = area_output_dir / sample_name
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    # webp is used for its compression properties of lossless images
    # webp can't handle 16bit channels so png is used instead for depth
    ext = "webp"
    image_rgb.save(sample_output_dir / f"rgb.{ext}", format=ext, lossless=True)
    depth.save(sample_output_dir / "depth.png", format="png", lossless=True)
    depth_mask.save(sample_output_dir / f"depth_mask.{ext}", format=ext, lossless=True)
    normals.save(sample_output_dir / f"normals.{ext}", format=ext, lossless=True)
    instances.save(sample_output_dir / f"instances.{ext}", format=ext, lossless=True)

    if save_depth_stats:
        # Compute statistics of the data
        masked_depth = np.ma.array(depth_m, mask=np.array(depth_mask) == 0)
        masked_depth = masked_depth.compressed()
        depth_hist, bin_edges = np.histogram(
            masked_depth, bins=128 * 10, range=(0, 128)
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
    print("Preprocessing Stanford-2D-3D Semantic dataset")

    parser = create_parser()
    args = parser.parse_args()
    # debug = args.debug

    if args.debug:
        print(f"-- Passed arguments: {args}")

        try:
            import matplotlib.pyplot as plt  # type:ignore
        except ImportError:
            import warnings

            warnings.warn("Matplotlib must be installed for plotting histograms!")
            print("Proceeding without plotting...")
            args.debug = False

    dataset_dir = Path(args.dataset_root)

    # Verify data validity
    dir_contents = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
    area_list = [d for d in dir_contents if "area_" in d]
    area_list.sort()  # Keep the areas sorted
    assert area_list, f"No areas were found in {dataset_dir}"
    assert "assets" in dir_contents, f"Missing 'assets' directory in {dataset_dir}"

    output_dir = Path(args.output_root)
    if not output_dir.is_dir():
        print(f"{output_dir} doesn't exist, creating...")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "assets").mkdir(parents=True, exist_ok=True)

    # Copy assets folder to output dir
    # TODO This will be added in future python versions
    # Maybe update when available to reduce dependencies
    # (dataset_dir / "assets").copy_into(output_dir, dirs_exist_ok=True)
    copytree(dataset_dir / "assets", output_dir / "assets", dirs_exist_ok=True)

    # Ensure resize ratio is within the specified limit
    scale = min(max(args.resize_image, 0.1), 1)

    cache_file = output_dir / "cache_samples_file_names.json"
    area_dict = {}
    # If the cache file exists and not to be rewritten, then load it
    if args.overwrite or not cache_file.exists():
        print("-- Reading file names: scanning files")
        for area in area_list:
            area_dict[area] = read_sample_names(dataset_dir / area / "pano" / "rgb")

        with open(cache_file, "w") as f:
            json.dump(area_dict, f)
    else:
        print("-- Reading file names: loading cached names")
        with open(cache_file, "r") as f:
            area_dict = json.load(f)

    depth_hist_dict = {}
    cache_file = output_dir / "cache_area_depth_statistics.json"
    # If the cache file exists and not to be rewritten, then don't do the computations
    print("-- Processing samples")
    save_depth_stats = args.overwrite or not cache_file.exists()
    for area, sample_list in area_dict.items():
        depth_hist_dict[area] = []
        area_output_dir = output_dir / area
        area_output_dir.mkdir(parents=True, exist_ok=True)
        for sample_name in tqdm(
            sample_list,
            desc=f"  -- Processing {area}",
            ncols=100,
            unit=" samples",
            # dynamic_ncols=True,
        ):
            sample_depth_hist = process_sample(
                dataset_dir / area / "pano",
                area_output_dir,
                sample_name,
                scale,
                save_depth_stats,
            )
            if save_depth_stats:
                try:
                    depth_hist_dict[area][0] += sample_depth_hist[0]
                    # Bins are the same no need to save them
                except IndexError:
                    depth_hist_dict[area] = sample_depth_hist

        if args.debug and save_depth_stats:
            plot_histogram(
                plt,  # type:ignore
                depth_hist_dict[area][0],
                depth_hist_dict[area][1][:-1],
                f"Histogram of depths in {area}",
            )

    if save_depth_stats:
        if args.debug:
            depth_hist = []
            for _, value in depth_hist_dict.items():
                if depth_hist:
                    depth_hist[0] += value[0]
                else:
                    depth_hist = value

            plot_histogram(
                plt,  # type:ignore
                depth_hist[0],
                depth_hist[1][:-1],
                "Histogram of depths in all areas",
            )

        print("-- Saving depth statistics")
        cache_dict = {}
        for key, value in depth_hist_dict.items():
            cache_dict[key] = [nparr.tolist() for nparr in value]

        with open(cache_file, "w") as f:
            json.dump(cache_dict, f)


if __name__ == "__main__":
    main()
