# PanoSAMic

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

PanoSAMic is a semantic segmentation model for panoramic images that integrates the pre-trained Segment Anything Model (SAM) encoder with multi-modal fusion capabilities. Existing image foundation models are not optimized for spherical images, having been trained primarily on perspective images. PanoSAMic addresses this by modifying the SAM encoder to output multi-stage features and introducing a novel spatio-modal fusion module that allows the model to select relevant modalities and features for different areas of the input.

Our semantic decoder uses spherical attention and dual view fusion to overcome the distortions and edge discontinuity often associated with panoramic images. PanoSAMic achieves state-of-the-art results on:
- **Stanford2D3DS**: RGB, RGB-D, and RGB-D-N modalities
- **Matterport3D**: RGB and RGB-D modalities

## Installation

1.  **Download Weights:**
    Please download the official pretrained weights from the [original SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints).

    * `sam_vit_h_4b8939.pth`
    * `sam_vit_l_0b3195.pth`
    * `sam_vit_b_01ec64.pth`

2. Clone the repository, install the depedencies, and link the weights dir

    ```shell
    $ git clone git@github.com:dfki-av/PanoSAMic.git

    $ cd PanoSAMic
    $ uv sync

    $ ln -s /path/to/sam/weights/* sam_weights/
    ```

Instead of linking, you can also directly put the weights in `sam_weights` directory.

## Usage

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
- `--dataset`: Choose from `stanford2d3ds`, `matterport3d`, or `structured3d`
- `--vit_model`: SAM encoder variant (`vit_h`, `vit_l`, or `vit_b`)
- `--modalities`: Comma-separated modalities (`image`, `depth`, `normals`)
- `--fold`: Dataset fold number for cross-validation
- `--resume`: Continue training from `last` or `best` checkpoint

### Evaluation

Evaluate a trained model:

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

### Configuration Files

Configuration files in the [config/](config/) directory control model architecture and training parameters. Available configs:
- `config_stanford2d3ds_dv.json` - Stanford2D3DS dual-view configuration
- `config_stanford2d3ds_sv.json` - Stanford2D3DS single-view configuration
- `config_matterport3d_dv.json` - Matterport3D dual-view configuration
- `config_baseline.json` - Baseline configuration

### SAM3 Baseline Evaluation

For comparison with SAM3 baselines, install the optional SAM3 dependencies:

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

The SAM3 checkpoint will be automatically downloaded from HuggingFace to your cache directory on first run.

## Data Preparation

### Dataset Downloads

Download the datasets from their respective sources:

* **Stanford-2D-3D-S**: [https://github.com/alexsax/2D-3D-Semantics](https://github.com/alexsax/2D-3D-Semantics)
* **Matterport-3D** (pre-processed 360FV-Matterport): [https://github.com/InSAI-Lab/360BEV](https://github.com/InSAI-Lab/360BEV)
* **Structured-3D**: [https://github.com/bertjiazheng/Structured3D](https://github.com/bertjiazheng/Structured3D)

After downloading the data from their respective sources, use the scripts in `panosamic/data_preparation/` to process them in the correct structure.

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

### Structured-3D
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
    2D_rendering/
        [sample_name]/
            panorama/
                full/
                    albedo.png
                    depth.png
                    instance.png
                    normal.png
                    rgb_coldlight.png
                    rgb_rawlight.png
                    rgb_warmlight.png
                    semantic.png
...
[scene_name]/
assets/
```
</td>
<td valign="top">

```scheme
[scene_name]/
    [sample_name]/
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

...

## License

This project is modfies parts of the **Segment Anything Model (SAM)**.

* **Original SAM Code:** Licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) by Meta AI.
* **Modified and Additional Components:** The modified encoder code in this repository is licensed under **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)** (Attribution-NonCommercial-ShareAlike).

### Model Weights
This code is designed to use the official pretrained SAM weights from Meta AI. The weights remain under their original [Apache 2.0 license](https://github.com/facebookresearch/segment-anything).