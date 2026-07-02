#!/usr/bin/env bash
# Portable UPAIR environment bootstrap.
# Source this file from any wrapper in the repository root.
#
# Important design point:
# - Default checks use Python package metadata only and do NOT import TensorFlow/Sionna.
#   TensorFlow/Sionna imports can stall for a long time on Compute Canada login nodes.
# - To explicitly run a deep import test, set UPAIR_DEEP_IMPORT_CHECK=1.
set -euo pipefail

export UPAIR_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UPAIR_REPO_PARENT="$(cd "${UPAIR_REPO_ROOT}/.." && pwd)"
# User-requested exact name/spelling: one folder above the repository.
export UPAIR_VENV_PATH="${UPAIR_VENV_PATH:-${UPAIR_REPO_PARENT}/.vevn_upair_potable}"
export UPAIR_REQUIREMENTS_FILE="${UPAIR_REQUIREMENTS_FILE:-${UPAIR_REPO_ROOT}/requirements-narval.txt}"
if [[ ! -f "${UPAIR_REQUIREMENTS_FILE}" ]]; then
  export UPAIR_REQUIREMENTS_FILE="${UPAIR_REPO_ROOT}/requirements.txt"
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${UPAIR_REPO_PARENT}/.cache/pip}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${UPAIR_REPO_PARENT}/.cache/cuda}"
mkdir -p "${PIP_CACHE_DIR}" "${CUDA_CACHE_PATH}"

_upair_module_cmd_available() {
  command -v module >/dev/null 2>&1 || [[ -n "${LMOD_CMD:-}" ]]
}

_upair_load_modules_if_requested() {
  if ! _upair_module_cmd_available; then
    return 0
  fi
  # Explicit override, e.g.:
  # export UPAIR_MODULES="StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9"
  if [[ -n "${UPAIR_MODULES:-}" ]]; then
    # shellcheck disable=SC2206
    local modules=( ${UPAIR_MODULES} )
    module load "${modules[@]}" || true
    return 0
  fi
  # Conservative default: only try Python. Do not force CUDA/cuDNN modules because
  # Compute Canada clusters differ and TensorFlow wheels may bring their own deps.
  if [[ "${UPAIR_AUTO_LOAD_PYTHON_MODULE:-1}" == "1" ]]; then
    module load StdEnv/2023 >/dev/null 2>&1 || true
    module load python/3.11 >/dev/null 2>&1 || module load python/3.10 >/dev/null 2>&1 || true
  fi
}

_upair_find_python() {
  if [[ -n "${UPAIR_PYTHON:-}" ]]; then
    command -v "${UPAIR_PYTHON}" >/dev/null 2>&1 || { echo "[ENV] UPAIR_PYTHON not found: ${UPAIR_PYTHON}" >&2; return 1; }
    echo "${UPAIR_PYTHON}"
    return 0
  fi
  local candidates=(python3.11 python3.10 python3 python)
  local py
  for py in "${candidates[@]}"; do
    if command -v "${py}" >/dev/null 2>&1; then
      if "${py}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
      then
        echo "${py}"
        return 0
      fi
    fi
  done
  echo "[ENV] Could not find Python >= 3.10. Load a Python module or set UPAIR_PYTHON." >&2
  return 1
}

_upair_verify_package_metadata() {
  python - <<'PY'
from __future__ import annotations
import sys
from importlib import metadata

if sys.version_info < (3, 10):
    print(f"[ENV] python>=3.10 required, got {sys.version.split()[0]}")
    raise SystemExit(1)

# Distribution names, not import names. This avoids importing TensorFlow/Sionna.
required = [
    "numpy",
    "pandas",
    "matplotlib",
    "PyYAML",
    "scipy",
    "tensorflow",
    "sionna-no-rt",
    "optuna",
]
missing = []
versions = {}
for package in required:
    try:
        versions[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        missing.append(package)

if missing:
    print("[ENV] Missing package metadata; requirements need installation:")
    for package in missing:
        print(f"  - {package}")
    raise SystemExit(1)

print(f"[ENV] Python {sys.version.split()[0]}")
for package in required:
    print(f"[ENV] {package} {versions[package]}")
PY
}

_upair_deep_verify_imports() {
  python - <<'PY'
from __future__ import annotations
import importlib
import time

required = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("yaml", "PyYAML"),
    ("scipy", "scipy"),
    ("tensorflow", "tensorflow"),
    ("sionna", "sionna-no-rt"),
    ("optuna", "optuna"),
]
for module_name, package_name in required:
    print(f"[ENV-DEEP] importing {module_name} ({package_name}) ...", flush=True)
    t0 = time.time()
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", "unknown")
    print(f"[ENV-DEEP] OK {module_name} {version} in {time.time() - t0:.2f}s", flush=True)
PY
}

upair_ensure_venv() {
  _upair_load_modules_if_requested

  local created=0
  if [[ ! -x "${UPAIR_VENV_PATH}/bin/python" ]]; then
    local py
    py="$(_upair_find_python)"
    echo "[ENV] Creating portable venv: ${UPAIR_VENV_PATH}"
    "${py}" -m venv "${UPAIR_VENV_PATH}"
    created=1
  else
    echo "[ENV] Reusing portable venv: ${UPAIR_VENV_PATH}"
  fi

  # shellcheck disable=SC1091
  source "${UPAIR_VENV_PATH}/bin/activate"
  export PATH="${UPAIR_VENV_PATH}/bin:${PATH}"
  export PYTHONPATH="${UPAIR_REPO_ROOT}/src:${UPAIR_REPO_ROOT}/scripts:${PYTHONPATH:-}"

  echo "[ENV] Checking installed package metadata only; TensorFlow/Sionna are not imported here."
  if [[ "${created}" == "1" ]] || ! _upair_verify_package_metadata >/dev/null 2>&1; then
    echo "[ENV] Installing/updating requirements from ${UPAIR_REQUIREMENTS_FILE}"
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r "${UPAIR_REQUIREMENTS_FILE}"
    python -m pip install --no-deps -e "${UPAIR_REPO_ROOT}"
  fi

  _upair_verify_package_metadata

  if [[ "${UPAIR_DEEP_IMPORT_CHECK:-0}" == "1" ]]; then
    echo "[ENV] Running optional deep import check. This imports TensorFlow/Sionna."
    if command -v timeout >/dev/null 2>&1; then
      export -f _upair_deep_verify_imports
      timeout "${UPAIR_DEEP_IMPORT_TIMEOUT:-180s}" bash -c '_upair_deep_verify_imports'
    else
      _upair_deep_verify_imports
    fi
  else
    echo "[ENV] Skipping TensorFlow/Sionna import check on this node. Set UPAIR_DEEP_IMPORT_CHECK=1 to test imports explicitly."
  fi

  echo "[ENV] Ready: REPO=${UPAIR_REPO_ROOT}"
  echo "[ENV] Ready: VENV=${UPAIR_VENV_PATH}"
}

upair_activate() {
  upair_ensure_venv
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  upair_ensure_venv
fi
