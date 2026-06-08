#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

task_name="${1:-${TASK_NAME:-place_dual_shoes}}"
setting="${2:-${TASK_CONFIG:-demo_clean}}"
expert_data_num="${3:-${EXPERT_DATA_NUM:-100}}"
seed="${4:-${SEED:-0}}"
gpu_id="${5:-${GPU_ID:-0}}"
batch_size="${6:-${BATCH_SIZE:-32}}"
num_epochs="${7:-${NUM_EPOCHS:-300}}"
checkpoint_every="${8:-${CHECKPOINT_EVERY:-100}}"
config_name="${CONFIG_NAME:-GAP}"
model_3d="${MODEL_3D:-pi3}"
observation_chunk="${OBSERVATION_CHUNK:-20}"
interval="${INTERVAL:-5}"
wandb_mode="${WANDB_MODE:-offline}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export WANDB_MODE="${wandb_mode}"

printf 'Training GAP policy\n'
printf '  task=%s setting=%s expert_data_num=%s seed=%s gpu=%s\n' "${task_name}" "${setting}" "${expert_data_num}" "${seed}" "${gpu_id}"
printf '  batch_size=%s num_epochs=%s checkpoint_every=%s\n' "${batch_size}" "${num_epochs}" "${checkpoint_every}"

python scripts/train.py \
    --config-name="${config_name}" \
    task_name="${task_name}" \
    setting="${setting}" \
    expert_data_num="${expert_data_num}" \
    training.seed="${seed}" \
    training.device="cuda:0" \
    dataloader.batch_size="${batch_size}" \
    val_dataloader.batch_size="${batch_size}" \
    training.num_epochs="${num_epochs}" \
    training.checkpoint_every="${checkpoint_every}" \
    logging.mode="${wandb_mode}" \
    model_3d="${model_3d}" \
    observation_chunk="${observation_chunk}" \
    interval="${interval}"
