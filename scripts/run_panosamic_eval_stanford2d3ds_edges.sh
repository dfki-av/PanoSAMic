#!/usr/bin/env bash
set -euo pipefail

# Evaluate edge-only IoU for single- and dual-view RGB-only checkpoints.
# Defaults can be overridden via env vars before running.

DATASET_PATH=${DATASET_PATH:-/data/Datasets/Stanford2D3DS/Stanford2D3D_noXYZ/processed}
EXPERIMENTS_PATH=${EXPERIMENTS_PATH:-experiments/}
SAM_WEIGHTS_PATH=${SAM_WEIGHTS_PATH:-sam_weights/sam_vit_h_4b8939.pth}
VIT_MODEL=${VIT_MODEL:-vit_h}
EDGE_SIDE_FRACS=${EDGE_SIDE_FRACS:-"0.25 0.15 0.05"} # per-side width fractions to keep
FOLDS=${FOLDS:-"1 2 3"}
BATCH_SIZE=${BATCH_SIZE:-1}
WORKERS=${WORKERS:-2}
OUTPUT_ROOT=${OUTPUT_ROOT:-experiments/panosamic_eval_edges}
CONFIG_PATH_DUAL=${CONFIG_PATH_DUAL:-config/config_stanford2d3ds_dv.json}   # dual-view fusion
CONFIG_PATH_SINGLE=${CONFIG_PATH_SINGLE:-config/config_stanford2d3ds_sv.json} # single-view
RUN_DUAL=${RUN_DUAL:-true}
RUN_SINGLE=${RUN_SINGLE:-true}
VIS_DIR=${VIS_DIR:-} # Optional visualization root; per-tag subfolders will be used.
REFINE=${REFINE:-true} # Apply instance refinement

echo "Dataset:      ${DATASET_PATH}"
echo "Experiments:  ${EXPERIMENTS_PATH}"
echo "SAM weights:  ${SAM_WEIGHTS_PATH}"
echo "Folds:        ${FOLDS}"
echo "Edge strips:  ${EDGE_SIDE_FRACS} (per side)"
echo "Output root:  ${OUTPUT_ROOT}"
echo "Vis dir:      ${VIS_DIR:-<none>}"

run_eval() {
  local tag=$1
  local config_path=$2
  local output_file=$3

  echo ">>> ${tag} | config: ${config_path} | output: ${output_file}"
  CMD=(python3 scripts/eval_panosamic_multifold.py
    --dataset_path "${DATASET_PATH}"
    --config_path "${config_path}"
    --experiments_path "${EXPERIMENTS_PATH}"
    --sam_weights_path "${SAM_WEIGHTS_PATH}"
    --dataset stanford2d3ds
    --vit_model "${VIT_MODEL}"
    --modalities "image"
    --batch_size "${BATCH_SIZE}"
    --workers_per_gpu "${WORKERS}"
    --output "${output_file}"
    --folds ${FOLDS}
    --edge-side-fracs ${EDGE_SIDE_FRACS}
  )
  [[ "${REFINE}" == "true" ]] && CMD+=(--refine)
  if [ -n "${VIS_DIR}" ]; then
    CMD+=(--save-dir "${VIS_DIR}/${tag}")
  fi
  "${CMD[@]}"
}

mkdir -p "$(dirname "${OUTPUT_ROOT}")"

if [[ "${RUN_DUAL}" == "true" ]]; then
  run_eval "dual_view" "${CONFIG_PATH_DUAL}" "${OUTPUT_ROOT}_dual.json"
fi

if [[ "${RUN_SINGLE}" == "true" ]]; then
  run_eval "single_view" "${CONFIG_PATH_SINGLE}" "${OUTPUT_ROOT}_single.json"
fi
