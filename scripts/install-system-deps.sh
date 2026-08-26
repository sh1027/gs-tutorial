#!/usr/bin/env bash
set -Eeuo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
    echo "This script supports Debian/Ubuntu systems with apt-get." >&2
    exit 1
fi

if (( EUID == 0 )); then
    APT_PREFIX=()
elif command -v sudo >/dev/null 2>&1; then
    APT_PREFIX=(sudo)
else
    echo "Root privileges are required. Install sudo or run this script as root." >&2
    exit 1
fi

if [[ -n "${PYTHON_VERSION:-}" ]]; then
    if [[ ! "${PYTHON_VERSION}" =~ ^3[.](10|11|12)$ ]]; then
        echo "PYTHON_VERSION must be 3.10, 3.11, or 3.12." >&2
        exit 2
    fi
    PYTHON_PACKAGE="python${PYTHON_VERSION}"
else
    PYTHON_PACKAGE="python3"
fi

echo "Installing system dependencies for gs-tutorial"
"${APT_PREFIX[@]}" apt-get update
"${APT_PREFIX[@]}" apt-get install -y \
    build-essential \
    ffmpeg \
    "${PYTHON_PACKAGE}-dev" \
    "${PYTHON_PACKAGE}-venv"

echo "System dependencies installed"
