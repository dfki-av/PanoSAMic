#!/usr/bin/env bash
set -euo pipefail

# Defaults (override via env vars)
DATASET_PATH=${DATASET_PATH:-/data/Datasets/Matterport3D/processed}
CONFIG_PATH=${CONFIG_PATH:-config/config_matterport3d_dv.json}
EXPERIMENTS_PATH=${EXPERIMENTS_PATH:-experiments/}
SAM_WEIGHTS_PATH=${SAM_WEIGHTS_PATH:-sam_weights/sam_vit_h_4b8939.pth}
OUTPUT=${OUTPUT:-experiments/panosamic_eval_matterport.json}
VIS_DIR=${VIS_DIR:-} # Leave empty to skip visualization
FOLDS=${FOLDS:-"1"}
VIT_MODEL=${VIT_MODEL:-vit_h}
MODALITIES=${MODALITIES:-image}
BATCH_SIZE=${BATCH_SIZE:-1}
WORKERS=${WORKERS:-2}
REFINE=${REFINE:-false}

echo "Dataset:      ${DATASET_PATH}"
echo "Config:       ${CONFIG_PATH}"
echo "Experiments:  ${EXPERIMENTS_PATH}"
echo "SAM weights:  ${SAM_WEIGHTS_PATH}"
echo "Output:       ${OUTPUT}"
echo "Folds:        ${FOLDS}"
echo "Modalities:   ${MODALITIES}"
echo "Refine:       ${REFINE}"
echo "Vis dir:      ${VIS_DIR:-<none>}"

CMD=(python3 scripts/eval_panosamic_multifold.py
  --dataset_path "${DATASET_PATH}"
  --config_path "${CONFIG_PATH}"
  --experiments_path "${EXPERIMENTS_PATH}"
  --sam_weights_path "${SAM_WEIGHTS_PATH}"
  --dataset matterport3d
  --vit_model "${VIT_MODEL}"
  --modalities "${MODALITIES}"
  --batch_size "${BATCH_SIZE}"
  --workers_per_gpu "${WORKERS}"
  --output "${OUTPUT}"
  --folds ${FOLDS}
)

if [ "${REFINE}" = "true" ]; then
  CMD+=(--refine)
fi
if [ -n "${VIS_DIR}" ]; then
  CMD+=(--save-dir "${VIS_DIR}")
fi

"${CMD[@]}"
