#!/usr/bin/env bash
set -euo pipefail

# Matterport3D validation split only (fold 1)
DATASET_PATH=${DATASET_PATH:-/data/Datasets/Matterport3D/processed}
SAM3_CKPT=${SAM3_CKPT:-${HOME}/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt}
OUTPUT=${OUTPUT:-runs/sam3_eval_matterport3d.json}
VIS_DIR=${VIS_DIR:-} # Leave empty to skip visualization
SMOOTH_KERNEL=${SMOOTH_KERNEL:-3}

echo "Dataset:   ${DATASET_PATH}"
echo "Checkpoint:${SAM3_CKPT}"
echo "Output:    ${OUTPUT}"
echo "Vis dir:   ${VIS_DIR:-<none>}"
echo "Smoothing: ${SMOOTH_KERNEL}"

CMD=(python3 scripts/eval_sam3_panosamic.py
    --dataset "fvmatterport3d"
    --dataset-path "${DATASET_PATH}"
    --checkpoint "${SAM3_CKPT}"
    --folds 1
    --output "${OUTPUT}"
    --confidence 0.25
    --coverage-threshold 0.05
    --smooth-kernel "${SMOOTH_KERNEL}"
)

if [ -n "${VIS_DIR}" ]; then
    CMD+=(--save-dir "${VIS_DIR}")
fi

"${CMD[@]}"
