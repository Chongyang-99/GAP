#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-}"
if [ -z "${ROBOTWIN_ROOT}" ] && [ -f "${SCRIPT_DIR}/../RoboTwin-cvpr-env/script/eval_policy.py" ]; then
    ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../RoboTwin-cvpr-env" && pwd)"
fi

task_name="${1:-${TASK_NAME:-place_dual_shoes}}"
task_config="${2:-${TASK_CONFIG:-demo_clean}}"
expert_data_num="${3:-${EXPERT_DATA_NUM:-100}}"
gpu_id="${4:-${GPU_ID:-0}}"
cameras="${CAMERAS:-head_camera}"
batch_size="${BATCH_SIZE:-256}"
observation_chunk="${OBSERVATION_CHUNK:-20}"
interval="${INTERVAL:-5}"
model_3d="${MODEL_3D:-pi3}"
raw_data_root="${RAW_DATA_ROOT:-${ROBOTWIN_ROOT:+${ROBOTWIN_ROOT}/data}}"
raw_data_root="${raw_data_root:-./data/raw}"
output_root="${OUTPUT_ROOT:-./data}"
pretrained_root="${GAP_PRETRAINED_ROOT:-pretrained}"
pi3_model_name_or_path="${PI3_MODEL_NAME_OR_PATH:-${PI3_MODEL_PATH:-${pretrained_root}/Pi3}}"
dinov3_repo_dir="${DINOV3_REPO_DIR:-thirdparty/dinov3}"
dinov3_weights_path="${DINOV3_WEIGHTS_PATH:-${pretrained_root}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"

printf 'Processing GAP data\n'
printf '  task=%s config=%s expert_data_num=%s gpu=%s\n' "${task_name}" "${task_config}" "${expert_data_num}" "${gpu_id}"
printf '  raw_data_root=%s output_root=%s\n' "${raw_data_root}" "${output_root}"

python scripts/process_data.py \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --cameras ${cameras} \
    --batch_size "${batch_size}" \
    --model_3d "${model_3d}" \
    --observation_chunk "${observation_chunk}" \
    --interval "${interval}" \
    --raw_data_root "${raw_data_root}" \
    --output_root "${output_root}" \
    --pi3_model_name_or_path "${pi3_model_name_or_path}" \
    --dinov3_repo_dir "${dinov3_repo_dir}" \
    --dinov3_weights_path "${dinov3_weights_path}"
