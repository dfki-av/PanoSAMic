# PanoSAMic

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![arXiv](https://img.shields.io/badge/arXiv-2601.07447-b31b1b.svg)](https://arxiv.org/abs/2601.07447)
[![HuggingFace](https://img.shields.io/badge/🤗-dfki--av%2FPanoSAMic-yellow)](https://huggingface.co/dfki-av/PanoSAMic)
[![Space](https://img.shields.io/badge/🤗%20Space-Demo-blue)](https://huggingface.co/spaces/dfki-av/PanoSAMic-demo)

PanoSAMic is a semantic segmentation model for panoramic images that integrates the pre-trained Segment Anything Model (SAM) encoder with multi-modal fusion capabilities. Existing image foundation models are not optimized for spherical images, having been trained primarily on perspective images. PanoSAMic addresses this by modifying the SAM encoder to output multi-stage features and introducing a novel spatio-modal fusion module that allows the model to select relevant modalities and features for different areas of the input.

Our semantic decoder uses spherical attention and dual view fusion to overcome the distortions and edge discontinuity often associated with panoramic images. PanoSAMic achieves state-of-the-art results on:
- **Stanford2D3DS**: RGB, RGB-D, and RGB-D-N modalities
- **Matterport3D**: RGB and RGB-D modalities

## Installation

**GPU requirements:** ≥16 GB VRAM for ViT-H inference · ≥24 GB for training · `scripts/demo.py` supports Apple Silicon (MPS) acceleration; the main training/evaluation pipeline (`train.py`/`evaluate.py`) has no MPS support and falls back to CPU without CUDA

1. Clone the repository and install dependencies:

    ```shell
    git clone git@github.com:dfki-av/PanoSAMic.git
    cd PanoSAMic
    uv sync
    ```

2. **SAM backbone weights** — choose one option:

    - **Auto-download (recommended):** pass `--sam_weights_path` to any script and
      the weights are fetched from Meta's servers on first use and cached under
      `~/.cache/panosamic/sam/`.

    - **Manual download:** grab the weights from the
      [SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints)
      and place or symlink them in `sam_weights/`:

      ```shell
      ln -s /path/to/sam/weights/* sam_weights/
      ```

## Usage

### Quick Demo

A hosted browser demo is available at
[huggingface.co/spaces/dfki-av/PanoSAMic-demo](https://huggingface.co/spaces/dfki-av/PanoSAMic-demo) —
no install required.

To run the model locally on any single panoramic image (no dataset download needed),
use `scripts/demo.py`. It writes a colorized segmentation image, the raw per-pixel
class-id array, and a class legend to an output directory:

```shell
python scripts/demo.py --image /path/to/panorama.jpg --output_dir demo_output/
```

By default this uses the released RGB-only Stanford2D3DS checkpoint (no depth/normals
needed). Pass `--checkpoint`, `--subfolder`, `--config_path`, and `--modalities` to use
a different released checkpoint (e.g. one trained with depth/normals).

### Training

Train PanoSAMic on a dataset using the training script:

```shell
python panosamic/evaluation/train.py \
    --dataset_path /path/to/processed/dataset \
    --config_path config/config_stanford2d3ds_dv.json \
    --experiments_path ./experiments \
    --sam_weights_path ./sam_weights \
    --dataset stanford2d3ds \
    --fold 1 \
    --batch_size 1 \
    --epochs 50 \
    --vit_model vit_h \
    --modalities image,depth,normals \
    --num_gpus 1 \
    --workers_per_gpu 2
```

**Key Parameters:**
- `--dataset`: Choose from `stanford2d3ds` or `matterport3d` (`tof360` is also
  supported, but for zero-shot evaluation only — see below)
- `--vit_model`: SAM encoder variant (`vit_h`, `vit_l`, or `vit_b`)
- `--modalities`: Comma-separated modalities (`image`, `depth`, `normals`)
- `--fold`: Dataset fold number for cross-validation
- `--resume`: Continue training from `last` or `best` checkpoint

### Evaluation

Evaluate a local training run (full checkpoint from `./experiments`):

```shell
python panosamic/evaluation/evaluate.py \
    --dataset_path /path/to/processed/dataset \
    --config_path config/config_stanford2d3ds_dv.json \
    --experiments_path ./experiments \
    --dataset stanford2d3ds \
    --fold 1 \
    --vit_model vit_h \
    --modalities image,depth,normals \
    --num_gpus 1 \
    --workers_per_gpu 2
```

### Evaluate from a released checkpoint

Reproduce paper results directly from the Hub (no local training run needed).
The frozen SAM backbone is fetched automatically if `--sam_weights_path` is
omitted:

```shell
python panosamic/evaluation/evaluate.py \
    --dataset_path /path/to/processed/dataset \
    --config_path config/config_stanford2d3ds_dv.json \
    --checkpoint dfki-av/PanoSAMic \
    --subfolder stanford2d3ds-vith-rgbdn-fold1 \
    --sam_weights_path ./sam_weights \
    --dataset stanford2d3ds \
    --fold 1 \
    --vit_model vit_h \
    --modalities image,depth,normals \
    --num_gpus 1
```

`--checkpoint` also accepts a local path to a `model.safetensors` file or a
directory containing one (e.g. exported via `scripts/export_checkpoint_for_hub.py`).

See [`MODEL_CARD.md`](MODEL_CARD.md) for the full checkpoint table and the
numbers each checkpoint reproduces.

### Zero-shot evaluation on ToF-360

We don't train on [ToF-360](https://huggingface.co/datasets/COLE-Ricoh/ToF-360)
(only 179 real-world samples across 4 scenes — too little to train on) — instead
we zero-shot evaluate a Stanford2D3DS-pretrained checkpoint against it, with
ToF-360's own label ontology remapped to Stanford2D3DS's 13 classes at load time
(see `panosamic/datasets/tof360.py`):

```shell
python panosamic/evaluation/evaluate.py \
    --dataset_path /path/to/processed/tof360 \
    --config_path config/config_tof360_dv.json \
    --checkpoint dfki-av/PanoSAMic \
    --subfolder stanford2d3ds-vith-rgb-fold1 \
    --sam_weights_path ./sam_weights \
    --dataset tof360 \
    --fold 1 \
    --vit_model vit_h \
    --modalities image \
    --num_gpus 1
```

### Configuration Files

Configuration files in the [config/](config/) directory control model architecture and training parameters. Available configs:
- `config_stanford2d3ds_dv.json` - Stanford2D3DS dual-view configuration
- `config_stanford2d3ds_sv.json` - Stanford2D3DS single-view configuration
- `config_matterport3d_dv.json` - Matterport3D dual-view configuration
- `config_tof360_dv.json` - ToF-360 zero-shot evaluation configuration (mirrors `config_stanford2d3ds_dv.json`)
- `config_baseline.json` - Baseline configuration

### SAM3 Baseline Evaluation

For comparison with SAM3 baselines, install the optional SAM3 dependency:

```shell
uv sync --extra sam3
```

Run SAM3 evaluation scripts:

```shell
# Stanford2D3DS evaluation
DATASET_PATH=/path/to/processed/dataset ./scripts/run_sam3_eval_stanford2d3ds.sh

# Matterport3D evaluation
DATASET_PATH=/path/to/processed/dataset ./scripts/run_sam3_eval_matterport3d.sh
```

The SAM3 model (`facebook/sam3`) is loaded via HuggingFace Transformers and downloaded automatically to your cache on first run.

## Development

### Running tests

```shell
# Full CPU test suite (no GPU required)
uv run pytest tests/

# Skip CUDA tests explicitly (e.g. when GPU is in use)
uv run pytest tests/ --ignore=tests/model/smoke/test_cuda.py \
    --ignore=tests/sam3/smoke/test_cuda.py \
    --ignore=tests/sam3/outputs/test_cuda.py

# Hub integration tests (downloads ~750 MB–1.5 GB from dfki-av/PanoSAMic)
PANOSAMIC_HUB_TESTS=1 uv run pytest tests/model/test_hub.py -v
```

Hub tests are skipped by default to avoid network I/O in regular runs.
Set `PANOSAMIC_HUB_TESTS=1` to verify that released checkpoints still load
correctly and contain no SAM backbone weights.
The checkpoint size reflects the trainable weights only (no SAM backbone):
~367 M parameters, ~740 MB in bfloat16 or ~1.5 GB in float32.

### Linting and type checking

```shell
uv run ruff check --fix   # lint with auto-fix
uv run ruff format        # format
uv run ty check           # type check
```

Pre-commit runs all three automatically on every commit.

## Data Preparation

### Dataset Downloads

Download the datasets from their respective sources:

* **Stanford-2D-3D-S**: [https://github.com/alexsax/2D-3D-Semantics](https://github.com/alexsax/2D-3D-Semantics)
* **Matterport-3D** (pre-processed 360FV-Matterport): [https://github.com/InSAI-Lab/360BEV](https://github.com/InSAI-Lab/360BEV)
* **ToF-360** (used for zero-shot evaluation only, see below): [https://huggingface.co/datasets/COLE-Ricoh/ToF-360](https://huggingface.co/datasets/COLE-Ricoh/ToF-360)

For convenience, the pre-processed Stanford-2D-3D-S and Matterport-3D datasets can be downloaded [here](https://cloud.dfki.de/owncloud/index.php/s/psKpwsL8B3yT3dK).

After downloading the data from their respective sources, use the scripts in `panosamic/data_preparation/` to process them in the correct structure.

Preprocessing code for Structured-3D also exists (`panosamic/data_preparation/structured3d.py`) but
isn't part of our reported results — we ultimately trained and evaluated on real captured data only.

### Stanford-2D-3D-S
<table width="100%">
<colgroup>
    <col style="width: 50%;">
</colgroup>

<tr>
<th><center>Original folder structure</th>
<th><center>Processed folder structure</th>
</tr>
<tr>
<td valign="top">

```scheme
area_1/
    pano/
        depth/
            [sample_name].png
        normal/
            [sample_name].png
        rgb/
            [sample_name].png
        semantic/
            [sample_name].png
area_2/
area_3/
area_4/
area_5a/
area_5b/
area_6/
assets/
```
</td>
<td valign="top">

```scheme
area_1/
    [sample_name]/
        depth.png
        depth_mask.webp
        instances.webp
        normals.webp
        rgb.webp
area_2/
area_3/
area_4/
area_5a/
area_5b/
area_6/
assets/
[cache_files]
```
</td>
</tr>
</table>

### Matterport-3D
<table width="100%">
<colgroup>
    <col style="width: 50%;">
</colgroup>

<tr>
<th><center>Original folder structure</th>
<th><center>Processed folder structure</th>
</tr>
<tr>
<td valign="top">

```scheme
[scene_name]/
    depth/
        [sample_name].png
    rgb/
        [sample_name].jpg
    semantic/
        [sample_name].png
...
[scene_name]/
```
</td>
<td valign="top">

```scheme
[scene_name]/
    [sample_name]/
        depth.png
        depth_mask.webp
        rgb.webp
        semantics.png
...
[scene_name]/
[cache_files]
```
</td>
</tr>
</table>

### ToF-360

Used for zero-shot evaluation only (see [above](#zero-shot-evaluation-on-tof-360)) —
we don't train on it. Only the `RGB/`, `depth/`, `normal/`, `semantics/`, and
`annotation/` subfolders are used (skipping `HHA/`, `XYZ/`, `RGB_mh_aligned/`,
`pretty/`, and `layout/`, which we don't need). ToF-360's raw semantic label
ids are copied through unchanged during preprocessing; the collapse to
Stanford2D3DS's 13-class taxonomy happens at load time via a hardcoded id
mapping in `panosamic/datasets/tof360.py`. Unlike the other three datasets,
ToF-360 ships no `assets/colors.npy` for visualization, so preprocessing
generates one (Stanford2D3DS's own published palette, since ToF360Dataset
reuses its exact class taxonomy).

<table width="100%">
<colgroup>
    <col style="width: 50%;">
</colgroup>

<tr>
<th><center>Original folder structure</th>
<th><center>Processed folder structure</th>
</tr>
<tr>
<td valign="top">

```scheme
[scene_name]/
    RGB/
        [idx]_[scene_name]_equi_rgb.png
    depth/
        [idx]_[scene_name]_equi_depth.png
    normal/
        [idx]_[scene_name]_equi_normal.png
    semantics/
        [idx]_[scene_name]_equi_semantic.npy
    annotation/
        [idx]_[scene_name]_equi_coco.json
...
[scene_name]/
```
</td>
<td valign="top">

```scheme
[scene_name]/
    [idx]/
        depth_mask.webp
        depth.png
        normals.webp
        rgb.webp
        semantics.png
...
[scene_name]/
assets/
[cache_files]
```
</td>
</tr>
</table>

## Citing this Work

```
@article{chamseddine2026panosamic,
    title   = {PanoSAMic: Panoramic Image Segmentation from SAM Feature Encoding and Dual View Fusion},
    author  = {Chamseddine, Mahdi and Stricker, Didier and Rambach, Jason},
    journal = {arXiv preprint arXiv:2601.07447},
    year    = {2026}
}
```

## Acknowledgement

This research was funded by the European Union as part of the projects: HumanTech (Grant Agreement 101058236) and ShieldBOT (Grant Agreement 101235093).

## License

This project is modfies parts of the **Segment Anything Model (SAM)**.

* **Original SAM Code:** Licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) by Meta AI.
* **Modified and Additional Components:** The modified encoder code in this repository is licensed under **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)** (Attribution-NonCommercial-ShareAlike).

### Model Weights
This code is designed to use the official pretrained SAM weights from Meta AI. The weights remain under their original [Apache 2.0 license](https://github.com/facebookresearch/segment-anything).
