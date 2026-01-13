#!/usr/bin/env bash
set -euo pipefail

# Default locations; override by exporting DATASET_PATH or SAM3_CKPT before running.
DATASET_PATH=${DATASET_PATH:-/data/Datasets/Stanford2D3DS/Stanford2D3D_noXYZ/processed}
SAM3_CKPT=${SAM3_CKPT:-${HOME}/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt}
OUTPUT=${OUTPUT:-runs/sam3_eval_stanford2d3ds.json}
VIS_DIR=${VIS_DIR:-runs/sam3_vis}

echo "Dataset:   ${DATASET_PATH}"
echo "Checkpoint:${SAM3_CKPT}"
echo "Output:    ${OUTPUT}"
echo "Vis dir:   ${VIS_DIR}"

python3 scripts/eval_sam3_panosamic.py \
    --dataset "stanford2d3ds" \
    --dataset-path "${DATASET_PATH}" \
    --checkpoint "${SAM3_CKPT}" \
    --folds 1 2 3 \
    --output "${OUTPUT}" \
    --save-dir "${VIS_DIR}" \
    --confidence 0.25 \
    --coverage-threshold 0.05 \
    --smooth-kernel 3
