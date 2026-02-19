#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[remote.sh] %s\n' "$*"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${ENV_FILE}"
  set +a
fi

CONFIG_REL_PATH="${CONFIG_REL_PATH:-configs/custom/eaton_palisades_seg_ft_remote.yaml}"
SMOKE_CONFIG_REL_PATH="${SMOKE_CONFIG_REL_PATH:-configs/custom/eaton_palisades_seg_smoke.yaml}"

DATASET_NAME="${DATASET_NAME:-eaton_palisades_30cm_4snap_temporal_refined_v1}"
DATASET_URL="${DATASET_URL:-https://huggingface.co/datasets/FuxunTB/wildfire/resolve/main/${DATASET_NAME}.zip}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"
DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT}/${DATASET_NAME}}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
HF_DATASET_REPO="${HF_DATASET_REPO:-FuxunTB/wildfire}"
HF_DATASET_FILE="${HF_DATASET_FILE:-${DATASET_NAME}.zip}"
HF_DATASET_REVISION="${HF_DATASET_REVISION:-main}"
HF_MODEL_REPO="${HF_MODEL_REPO:-facebook/sam3}"
HF_MODEL_FILE="${HF_MODEL_FILE:-sam3.pt}"
HF_MODEL_REVISION="${HF_MODEL_REVISION:-main}"
ALLOW_DATASET_DOWNLOAD="${ALLOW_DATASET_DOWNLOAD:-1}"

RUN_SMOKE="${RUN_SMOKE:-0}"              # 1 to run smoke config before full train
RUN_TRAIN="${RUN_TRAIN:-1}"              # 1 to run full train
NUM_GPUS="${NUM_GPUS:-auto}"             # set explicit integer to override auto-detect
USE_CLUSTER="${USE_CLUSTER:-0}"
UV_SYNC_ARGS="${UV_SYNC_ARGS:---extra train}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}" # e.g. "--partition xxx --account yyy"
SAM3_CKPT_PATH="${SAM3_CKPT_PATH:-}"     # optional: use existing local checkpoint path
APPLY_EATON_CLASS_NAME_TRICK="${APPLY_EATON_CLASS_NAME_TRICK:-1}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

ensure_system_deps() {
  if ! command -v apt-get >/dev/null 2>&1; then
    log "apt-get not found; skipping system package install."
    return
  fi

  local sudo_cmd=""
  if [[ "${EUID}" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      sudo_cmd="sudo"
    else
      log "Not root and sudo missing; skipping apt install."
      return
    fi
  fi

  log "Installing system packages required for setup/build."
  ${sudo_cmd} apt-get update -y
  DEBIAN_FRONTEND=noninteractive ${sudo_cmd} apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    unzip \
    wget \
    build-essential \
    pkg-config \
    python3-dev
}

ensure_uv() {
  export PATH="${HOME}/.local/bin:${PATH}"
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  log "Installing uv."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
  if ! command -v uv >/dev/null 2>&1; then
    log "uv install failed."
    exit 1
  fi
}

check_gpu_runtime() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi not found. GPU runtime may be unavailable."
    return
  fi

  if ! nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi exists but is not usable. Check NVIDIA driver/runtime."
    exit 1
  fi
}

check_repo_integrity() {
  local required=(
    "${PROJECT_DIR}/sam3/train/data/__init__.py"
    "${PROJECT_DIR}/sam3/train/data/collator.py"
    "${PROJECT_DIR}/sam3/model/sam3_image.py"
  )
  for p in "${required[@]}"; do
    if [[ ! -f "${p}" ]]; then
      log "Required source file missing: ${p}"
      log "Likely rsync exclude pattern issue. Ensure '/data/' is excluded, not all 'data/' paths."
      exit 1
    fi
  done
}

setup_python_env() {
  log "Syncing Python environment."
  cd "${PROJECT_DIR}"
  uv venv
  # shellcheck disable=SC2086
  uv sync ${UV_SYNC_ARGS}
  # Needed by COCO loaders during training.
  uv pip install --python .venv/bin/python pycocotools
}

download_dataset_archive() {
  local zip_path="$1"
  local auth_msg=""
  if [[ -n "${HF_TOKEN}" ]]; then
    auth_msg=" (with HF token)"
  fi
  log "Downloading dataset archive${auth_msg}."

  if _download_dataset_archive_direct "${zip_path}"; then
    return 0
  fi

  log "Direct URL download failed; trying Hugging Face Hub API."
  if _download_dataset_archive_via_hf_hub "${zip_path}"; then
    return 0
  fi

  return 1
}

_download_dataset_archive_direct() {
  local zip_path="$1"
  if command -v curl >/dev/null 2>&1; then
    if [[ -n "${HF_TOKEN}" ]]; then
      curl -fL -H "Authorization: Bearer ${HF_TOKEN}" -o "${zip_path}" "${DATASET_URL}"
    else
      curl -fL -o "${zip_path}" "${DATASET_URL}"
    fi
    return 0
  fi

  if [[ -n "${HF_TOKEN}" ]]; then
    wget --header="Authorization: Bearer ${HF_TOKEN}" -O "${zip_path}" "${DATASET_URL}"
  else
    wget -O "${zip_path}" "${DATASET_URL}"
  fi
}

_download_dataset_archive_via_hf_hub() {
  local zip_path="$1"
  uv run python - <<'PY' "${zip_path}" "${HF_DATASET_REPO}" "${HF_DATASET_FILE}" "${HF_DATASET_REVISION}" "${HF_TOKEN}"
import pathlib
import shutil
import sys

from huggingface_hub import hf_hub_download

zip_path, repo_id, filename, revision, token = sys.argv[1:6]
token = token or None
local_path = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename=filename,
    revision=revision,
    token=token,
)
dst = pathlib.Path(zip_path)
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(local_path, dst)
print(f"downloaded {dst}")
PY
}

prepare_dataset() {
  mkdir -p "${DATA_ROOT}"
  if [[ ! -d "${DATASET_ROOT}" ]]; then
    if [[ -n "${HF_TOKEN}" ]]; then
      log "HF token detected: yes"
    else
      log "HF token detected: no"
    fi
    local zip_path="${DATA_ROOT}/${DATASET_NAME}.zip"
    if [[ ! -f "${zip_path}" ]]; then
      if [[ "${ALLOW_DATASET_DOWNLOAD}" != "1" ]]; then
        log "Dataset missing at ${DATASET_ROOT} and download disabled (ALLOW_DATASET_DOWNLOAD=${ALLOW_DATASET_DOWNLOAD})."
        log "Rsync data/ to the node, or set ALLOW_DATASET_DOWNLOAD=1."
        exit 1
      fi
      if ! download_dataset_archive "${zip_path}"; then
        log "Dataset download failed from ${DATASET_URL}."
        log "Tried direct URL and hf_hub API for repo ${HF_DATASET_REPO}, file ${HF_DATASET_FILE}."
        log "If this is private/gated, ensure HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) is set on the remote shell and has dataset read access."
        exit 1
      fi
    else
      log "Found existing dataset archive: ${zip_path}"
    fi
    log "Extracting dataset archive."
    unzip -o "${zip_path}" -d "${DATA_ROOT}"
  else
    log "Dataset already exists at ${DATASET_ROOT}; skipping download."
  fi

  if [[ ! -d "${DATASET_ROOT}" ]]; then
    log "Expected dataset root not found: ${DATASET_ROOT}"
    exit 1
  fi
  if [[ ! -f "${DATASET_ROOT}/annotations/instances_train.json" ]]; then
    log "Missing annotations at ${DATASET_ROOT}/annotations/instances_train.json"
    exit 1
  fi

  # Ensure per-split annotation links expected by config.
  for split in train val test; do
    local ann_src="../annotations/instances_${split}.json"
    local ann_dst="${DATASET_ROOT}/${split}/_annotations.coco.json"
    mkdir -p "${DATASET_ROOT}/${split}"
    rm -f "${ann_dst}"
    ln -s "${ann_src}" "${ann_dst}"
  done
}

apply_annotation_class_name_trick() {
  if [[ "${APPLY_EATON_CLASS_NAME_TRICK}" != "1" ]]; then
    log "Skipping class-name text rewrite."
    return
  fi

  log "Applying class-name text rewrite for Eaton categories."
  uv run python - <<'PY' "${DATASET_ROOT}"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
ann_dir = root / "annotations"
ann_paths = sorted(ann_dir.glob("instances_*.json"))
if not ann_paths:
    raise SystemExit(f"No annotation files found under {ann_dir}")

name_map = {
    1: "building with no damage",
    2: "damaged building little",
    3: "damaged building minor",
    4: "damaged building medium",
    5: "destroyed building",
    6: "debris_cleared",
}

for path in ann_paths:
    data = json.loads(path.read_text())
    changed = False
    for cat in data.get("categories", []):
        cat_id = int(cat["id"])
        target = name_map.get(cat_id)
        if target is not None and cat.get("name") != target:
            cat["name"] = target
            changed = True
    if changed:
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"updated {path}")
    else:
        print(f"ok {path}")
PY
}

resolve_pretrained_checkpoint() {
  log "Resolving pretrained SAM3 checkpoint."
  local ckpt
  if [[ -n "${SAM3_CKPT_PATH}" ]]; then
    ckpt="${SAM3_CKPT_PATH}"
  else
    ckpt="$(
      uv run python - <<'PY' "${HF_MODEL_REPO}" "${HF_MODEL_FILE}" "${HF_MODEL_REVISION}" "${HF_TOKEN}" | tail -n 1
from huggingface_hub import hf_hub_download
import sys

repo_id, filename, revision, token = sys.argv[1:5]
token = token or None
path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    revision=revision,
    token=token,
)
print(path)
PY
    )"
  fi

  if [[ -z "${ckpt}" || ! -e "${ckpt}" ]]; then
    log "Failed to resolve pretrained checkpoint path."
    exit 1
  fi
  log "Using checkpoint: ${ckpt}"

  local cfg_ft="${PROJECT_DIR}/sam3/train/${CONFIG_REL_PATH}"
  local cfg_smoke="${PROJECT_DIR}/sam3/train/${SMOKE_CONFIG_REL_PATH}"

  # Patch config(s) so checkpoint path is valid on this machine.
  uv run python - <<'PY' "${ckpt}" "${cfg_ft}" "${cfg_smoke}"
import pathlib
import re
import sys

ckpt_path = sys.argv[1]
cfg_paths = [pathlib.Path(p) for p in sys.argv[2:] if pathlib.Path(p).exists()]

for cfg in cfg_paths:
    text = cfg.read_text()
    text = re.sub(
        r"(?m)^(\s*sam3_checkpoint_path:\s*).*$",
        rf"\1{ckpt_path}",
        text,
    )
    text = re.sub(r"(?mi)^(\s*load_from_HF:\s*).*$", r"\1false", text)
    cfg.write_text(text)
    print(f"patched {cfg}")
PY
}

run_train() {
  cd "${PROJECT_DIR}"
  if [[ "${NUM_GPUS}" == "auto" ]]; then
    local detected_gpus
    detected_gpus="$(
      uv run python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
    )"
    detected_gpus="${detected_gpus//[[:space:]]/}"
    if [[ -z "${detected_gpus}" || "${detected_gpus}" == "0" ]]; then
      log "Auto-detect found 0 visible GPUs. Set NUM_GPUS manually if needed."
      exit 1
    fi
    NUM_GPUS="${detected_gpus}"
    log "Auto-detected NUM_GPUS=${NUM_GPUS}"
  fi

  local base_cmd=(uv run python sam3/train/train.py --use-cluster "${USE_CLUSTER}" --num-gpus "${NUM_GPUS}")
  local smoke_cfg="${SMOKE_CONFIG_REL_PATH}"
  local ft_cfg="${CONFIG_REL_PATH}"

  if [[ ! -f "sam3/train/${ft_cfg}" ]]; then
    log "Training config not found: sam3/train/${ft_cfg}"
    exit 1
  fi

  if [[ "${RUN_SMOKE}" == "1" ]]; then
    if [[ ! -f "sam3/train/${smoke_cfg}" ]]; then
      log "Smoke config not found: sam3/train/${smoke_cfg}"
      exit 1
    fi
    log "Running smoke training: ${smoke_cfg}"
    "${base_cmd[@]}" -c "${smoke_cfg}"
  fi

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    log "Running full training: ${ft_cfg}"
    if [[ -n "${EXTRA_TRAIN_ARGS}" ]]; then
      # shellcheck disable=SC2206
      local extra_args=( ${EXTRA_TRAIN_ARGS} )
      "${base_cmd[@]}" -c "${ft_cfg}" "${extra_args[@]}"
    else
      "${base_cmd[@]}" -c "${ft_cfg}"
    fi
  fi
}

main() {
  log "Project dir: ${PROJECT_DIR}"
  ensure_system_deps
  ensure_uv
  check_repo_integrity
  check_gpu_runtime
  setup_python_env
  prepare_dataset
  apply_annotation_class_name_trick
  resolve_pretrained_checkpoint
  run_train
  log "Done."
}

main "$@"
