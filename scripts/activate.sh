#!/usr/bin/env bash
# Source this file so CUDA and the project virtual environment are configured
# in the current shell: source scripts/activate.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This script must be sourced: source scripts/activate.sh" >&2
    exit 2
fi

_gsplat_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export GSPLAT_TUTORIAL_ROOT="$(cd -- "${_gsplat_script_dir}/.." && pwd)"
export GSPLAT_VENV_DIR="${GSPLAT_VENV_DIR:-${GSPLAT_TUTORIAL_ROOT}/.venv}"

if [[ ! -x "${GSPLAT_VENV_DIR}/bin/python" ]]; then
    echo "Virtual environment not found: ${GSPLAT_VENV_DIR}" >&2
    echo "Run bash scripts/setup.sh first." >&2
    unset _gsplat_script_dir
    return 1
fi

if [[ -f "${GSPLAT_VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${GSPLAT_VENV_DIR}/bin/activate"
else
    # An interrupted `python -m venv` can leave a usable interpreter without
    # activation helpers. setup.sh repairs pip; this fallback makes that venv
    # usable without deleting and recreating it.
    export VIRTUAL_ENV="${GSPLAT_VENV_DIR}"
    unset PYTHONHOME 2>/dev/null || true
    export PATH="${VIRTUAL_ENV}/bin:${PATH}"
    hash -r
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
    if [[ -n "${GSPLAT_CUDA_HOME:-}" ]]; then
        export CUDA_HOME="${GSPLAT_CUDA_HOME}"
    elif [[ -x /usr/local/cuda-12.8/bin/nvcc ]]; then
        export CUDA_HOME=/usr/local/cuda-12.8
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        export CUDA_HOME=/usr/local/cuda
    elif command -v nvcc >/dev/null 2>&1; then
        export CUDA_HOME="$(cd -- "$(dirname -- "$(command -v nvcc)")/.." && pwd)"
    fi
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
else
    echo "Warning: CUDA Toolkit was not found; gsplat CUDA kernels cannot be compiled." >&2
fi

export MAX_JOBS="${MAX_JOBS:-8}"
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    _gsplat_detected_arch="$(python -c \
        'import torch; print(".".join(map(str, torch.cuda.get_device_capability()))) if torch.cuda.is_available() else print("")' \
        2>/dev/null || true)"
    # 8.9 is the architecture used for the input.MOV verification. A visible
    # GPU takes precedence, so other machines automatically use their own arch.
    export TORCH_CUDA_ARCH_LIST="${_gsplat_detected_arch:-${GSPLAT_TORCH_ARCH:-8.9}}"
    unset _gsplat_detected_arch
fi

unset _gsplat_script_dir
