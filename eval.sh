#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAP_ROOT="${SCRIPT_DIR}"
GAP_PARENT="$(cd "${GAP_ROOT}/.." && pwd)"

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-}"
if [ -z "${ROBOTWIN_ROOT}" ] && [ -f "${GAP_ROOT}/../RoboTwin-cvpr-env/script/eval_policy.py" ]; then
    ROBOTWIN_ROOT="$(cd "${GAP_ROOT}/../RoboTwin-cvpr-env" && pwd)"
fi

policy_name="${POLICY_NAME:-GAP}"
task_name="${1:-${TASK_NAME:-place_dual_shoes}}"
task_config="${2:-${TASK_CONFIG:-demo_clean}}"
ckpt_setting="${3:-${CKPT_SETTING:-${task_config}}}"
expert_data_num="${4:-${EXPERT_DATA_NUM:-100}}"
checkpoint_num="${5:-${CHECKPOINT_NUM:-300}}"
gpu_id="${6:-${GPU_ID:-0}}"
seed_list="${7:-${SEEDS:-0}}"
ckpt_path="${CKPT_PATH:-checkpoints/${task_name}_${ckpt_setting}_${expert_data_num}/${checkpoint_num}.ckpt}"
results_root="${RESULTS_ROOT:-${GAP_ROOT}/results}"

if [ ! -f "${ROBOTWIN_ROOT}/script/eval_policy.py" ]; then
    printf 'ROBOTWIN_ROOT must contain script/eval_policy.py: %s\n' "${ROBOTWIN_ROOT}" >&2
    exit 1
fi

if [[ "${ckpt_path}" = /* ]]; then
    checkpoint_file="${ckpt_path}"
elif [ -f "${GAP_ROOT}/${ckpt_path}" ]; then
    checkpoint_file="${GAP_ROOT}/${ckpt_path}"
elif [ -f "${ROBOTWIN_ROOT}/${ckpt_path}" ]; then
    checkpoint_file="${ROBOTWIN_ROOT}/${ckpt_path}"
elif [ -f "${ROBOTWIN_ROOT}/policy/${policy_name}/${ckpt_path}" ]; then
    checkpoint_file="${ROBOTWIN_ROOT}/policy/${policy_name}/${ckpt_path}"
else
    checkpoint_file="${GAP_ROOT}/${ckpt_path}"
fi

if [ ! -f "${checkpoint_file}" ]; then
    printf 'Checkpoint not found: %s\n' "${checkpoint_file}" >&2
    exit 1
fi

read -r -a seeds <<< "${seed_list}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${GAP_PARENT}:${PYTHONPATH:-}"
export EVAL_RESULT_ROOT="${results_root}"
mkdir -p "${EVAL_RESULT_ROOT}"

cd "${ROBOTWIN_ROOT}"

printf 'Evaluating GAP policy\n'
printf '  gap_root=%s robotwin_root=%s\n' "${GAP_ROOT}" "${ROBOTWIN_ROOT}"
printf '  results_root=%s\n' "${results_root}"
printf '  task=%s config=%s checkpoint=%s gpu=%s seeds=%s\n' "${task_name}" "${task_config}" "${checkpoint_file}" "${gpu_id}" "${seed_list}"

for seed in "${seeds[@]}"; do
    PYTHONWARNINGS=ignore::UserWarning \
    python "${GAP_ROOT}/scripts/eval_policy.py" --config "${GAP_ROOT}/deploy_policy.yml" \
        --overrides \
        --policy_name "${policy_name}" \
        --task_name "${task_name}" \
        --task_config "${task_config}" \
        --ckpt_setting "${ckpt_setting}" \
        --expert_data_num "${expert_data_num}" \
        --seed "${seed}" \
        --checkpoint_num "${checkpoint_num}" \
        --ckpt_path "${checkpoint_file}"

done
