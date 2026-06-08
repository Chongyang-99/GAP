#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PI3_REPO="${PI3_REPO:-yyfz233/Pi3}"
DINOV3_REPO="${DINOV3_REPO:-PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m}"
DINOV3_FILENAME="${DINOV3_FILENAME:-dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"
DINOV3_WEIGHTS_URL="${DINOV3_WEIGHTS_URL:-}"

require_python_package() {
    python - "$1" <<'PY'
import importlib.util
import sys

package_name = sys.argv[1]
if importlib.util.find_spec(package_name) is None:
    raise SystemExit(
        f"Missing Python package: {package_name}\n"
        f"Install it with: pip install {package_name.replace('_', '-')}"
    )
PY
}

download_pi3() {
    if [ -f "Pi3/config.json" ] && { [ -f "Pi3/model.safetensors" ] || [ -f "Pi3/pytorch_model.bin" ]; }; then
        printf 'Pi3 weights already exist at %s\n' "${SCRIPT_DIR}/Pi3"
        return
    fi

    require_python_package huggingface_hub
    python - "${PI3_REPO}" "${SCRIPT_DIR}/Pi3" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    token=os.environ.get("HF_TOKEN"),
)
print(f"Downloaded {repo_id} to {local_dir}")
PY
}

download_dinov3() {
    if [ -f "${DINOV3_FILENAME}" ]; then
        printf 'DINOv3 weights already exist at %s\n' "${SCRIPT_DIR}/${DINOV3_FILENAME}"
        return
    fi

    if [ -n "${DINOV3_WEIGHTS_URL}" ]; then
        if command -v curl >/dev/null 2>&1; then
            curl -L --fail "${DINOV3_WEIGHTS_URL}" -o "${DINOV3_FILENAME}"
        elif command -v wget >/dev/null 2>&1; then
            wget -O "${DINOV3_FILENAME}" "${DINOV3_WEIGHTS_URL}"
        else
            printf 'curl or wget is required for DINOV3_WEIGHTS_URL downloads.\n' >&2
            exit 1
        fi
        return
    fi

    require_python_package huggingface_hub
    python - "${DINOV3_REPO}" "${DINOV3_FILENAME}" "${SCRIPT_DIR}" <<'PY'
import os
import sys
from huggingface_hub import hf_hub_download

repo_id, filename, local_dir = sys.argv[1], sys.argv[2], sys.argv[3]
hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir,
    token=os.environ.get("HF_TOKEN"),
)
print(f"Downloaded {filename} from {repo_id} to {local_dir}")
PY
}

download_pi3
download_dinov3

printf '\nWeights are ready:\n'
printf '  %s\n' "${SCRIPT_DIR}/Pi3"
printf '  %s\n' "${SCRIPT_DIR}/${DINOV3_FILENAME}"
