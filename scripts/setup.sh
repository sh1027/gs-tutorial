#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${VENV_DIR:-${REPO_ROOT}/.venv}"
if [[ -n "${PYTHON_VERSION:-}" ]]; then
    if [[ ! "${PYTHON_VERSION}" =~ ^3[.](10|11|12)$ ]]; then
        echo "PYTHON_VERSION must be 3.10, 3.11, or 3.12." >&2
        exit 2
    fi
    PYTHON_COMMAND="python${PYTHON_VERSION}"
else
    PYTHON_COMMAND="python3"
fi
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
CHECK_ONLY=0

usage() {
    cat <<'EOF'
Usage: bash scripts/setup.sh [--check]

Create the local Python/CUDA environment used for the input.MOV verification.

Options:
  --check  Verify an existing environment without installing anything.
  -h       Show this help.

Optional environment variables:
  PYTHON_VERSION          Python version: 3.10, 3.11, or 3.12 (default: system python3)
  VENV_DIR               Virtual environment (default: <repo>/.venv)
  CUDA_HOME              CUDA Toolkit directory (default: auto-detect)
  TORCH_CUDA_ARCH_LIST    CUDA architectures (default: detected GPU, otherwise 8.9)
  TORCH_VERSION          PyTorch version (default: 2.5.1)
  TORCHVISION_VERSION    TorchVision version (default: 0.20.1)
  TORCH_INDEX_URL        PyTorch wheel index (default: cu121)
  MAX_JOBS               CUDA extension build parallelism (default: 8)
EOF
}

case "${1:-}" in
    "") ;;
    --check) CHECK_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac

if ! command -v "${PYTHON_COMMAND}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_COMMAND}" >&2
    exit 1
fi

"${PYTHON_COMMAND}" - <<'PY'
import sys

if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(f"Python 3.10-3.12 is required; found {sys.version.split()[0]}")
print(f"Python: {sys.version.split()[0]}")
PY

if (( CHECK_ONLY == 0 )); then
    if [[ ! -d "${VENV_PATH}" ]]; then
        echo "Creating virtual environment: ${VENV_PATH}"
        if command -v uv >/dev/null 2>&1; then
            uv venv --python "${PYTHON_COMMAND}" "${VENV_PATH}"
        elif ! "${PYTHON_COMMAND}" -m venv "${VENV_PATH}"; then
            echo "Could not create a complete virtual environment." >&2
            echo "Install python3-venv (or uv), then run this script again." >&2
            exit 1
        fi
    elif [[ ! -x "${VENV_PATH}/bin/python" ]]; then
        echo "Path exists but is not a Python virtual environment: ${VENV_PATH}" >&2
        exit 1
    fi

    if ! "${VENV_PATH}/bin/python" -m pip --version >/dev/null 2>&1; then
        echo "pip is missing from the virtual environment; bootstrapping it"
        if "${VENV_PATH}/bin/python" -m ensurepip --upgrade >/dev/null 2>&1; then
            :
        elif command -v uv >/dev/null 2>&1; then
            uv pip install --python "${VENV_PATH}/bin/python" pip setuptools wheel
        else
            echo "pip could not be bootstrapped because ensurepip and uv are unavailable." >&2
            echo "Install python3-venv (or uv), then run this script again." >&2
            exit 1
        fi
    fi
    if ! "${VENV_PATH}/bin/python" -m pip --version >/dev/null 2>&1; then
        echo "pip bootstrap failed for ${VENV_PATH}" >&2
        exit 1
    fi

    echo "Installing PyTorch ${TORCH_VERSION} from ${TORCH_INDEX_URL}"
    "${VENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel
    "${VENV_PATH}/bin/python" -m pip install \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url "${TORCH_INDEX_URL}"

    echo "Installing the tutorial and all local development extras"
    "${VENV_PATH}/bin/python" -m pip install -e \
        "${REPO_ROOT}[reconstruction,gpu,notebook,viewer,dev]"

    echo "Registering the Jupyter kernel: Python (gs-tutorial)"
    "${VENV_PATH}/bin/python" -m ipykernel install \
        --user \
        --name gs-tutorial \
        --display-name "Python (gs-tutorial)"
elif [[ ! -x "${VENV_PATH}/bin/python" ]]; then
    echo "Virtual environment not found: ${VENV_PATH}" >&2
    exit 1
elif ! "${VENV_PATH}/bin/python" -m pip --version >/dev/null 2>&1; then
    echo "pip is missing from ${VENV_PATH}; rerun without --check to repair it." >&2
    exit 1
fi

export VENV_DIR="${VENV_PATH}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/activate.sh"

python - <<'PY'
from importlib.metadata import version
import shutil

import torch

packages = ("torch", "gsplat", "pycolmap", "viser", "jupyterlab", "ipykernel", "pytest", "ruff")
for package in packages:
    print(f"{package}: {version(package)}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {'.'.join(map(str, torch.cuda.get_device_capability(0)))}")
for command in ("ffmpeg", "c++", "nvcc"):
    print(f"{command}: {shutil.which(command) or 'not found'}")
PY

echo
echo "Environment check completed. Activate it in each new terminal with:"
echo "  source scripts/activate.sh"
echo "Then run:"
echo "  gs-tutorial env"
