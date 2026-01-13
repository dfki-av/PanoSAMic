#!/bin/bash

# Example evaluation script - Update paths to match your setup
# You can override these by setting environment variables before running:
#   DATASET_PATH=/your/path/to/dataset ./scripts/local_eval.sh

DATASET_PATH=${DATASET_PATH:-/path/to/processed/dataset}
CONFIG_PATH=${CONFIG_PATH:-"$(pwd)"/config/config_stanford2d3ds_dv.json}
EXPERIMENTS_PATH=${EXPERIMENTS_PATH:-"$(pwd)"/experiments}

python panosamic/evaluation/evaluate.py \
    --dataset_path $DATASET_PATH \
    --config_path $CONFIG_PATH \
    --experiments_path $EXPERIMENTS_PATH \
    --num_gpus 1 \
    --workers_per_gpu 2\
    --dataset "stanford2d3ds" \
    --fold 1 \
    --vit_model "vit_h" \
    --modalities "image" \
