#!/usr/bin/env bash
set -euo pipefail

# Default locations; override by exporting DATASET_PATH before running.
DATASET_PATH=${DATASET_PATH:-/data/Datasets/Stanford2D3DS/Stanford2D3D_noXYZ/processed}
OUTPUT=${OUTPUT:-runs/sam3_eval_stanford2d3ds.json}
VIS_DIR=${VIS_DIR:-runs/sam3_vis}

echo "Dataset:   ${DATASET_PATH}"
echo "Output:    ${OUTPUT}"
echo "Vis dir:   ${VIS_DIR}"

python3 scripts/eval_sam3_panosamic.py \
    --dataset "stanford2d3ds" \
    --dataset-path "${DATASET_PATH}" \
    --folds 1 2 3 \
    --output "${OUTPUT}" \
    --save-dir "${VIS_DIR}" \
    --confidence 0.25 \
    --coverage-threshold 0.05 \
    --smooth-kernel 3
