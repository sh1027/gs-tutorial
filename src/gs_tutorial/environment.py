from __future__ import annotations

import importlib.metadata
import importlib.util
import shutil
import subprocess
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Check:
    name: str
    available: bool
    detail: str
    required_for: str


def _package_check(import_name: str, distribution: str, required_for: str) -> Check:
    available = importlib.util.find_spec(import_name) is not None
    try:
        detail = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        detail = "not installed"
    return Check(distribution, available, detail, required_for)


def collect_environment() -> list[Check]:
    """Return checks without importing CUDA extensions."""
    checks = [
        _package_check("numpy", "numpy", "all Python stages"),
        _package_check("PIL", "Pillow", "image loading"),
        _package_check("pycolmap", "pycolmap", "SfM when COLMAP CLI is absent"),
        _package_check("torch", "torch", "training"),
        _package_check("gsplat", "gsplat", "training and export"),
        _package_check("viser", "viser", "interactive web viewer"),
        _package_check("jupyterlab", "jupyterlab", "notebooks"),
    ]
    for command, purpose in [("ffmpeg", "video input"), ("colmap", "SfM CLI backend")]:
        path = shutil.which(command)
        checks.append(Check(command, path is not None, path or "not found", purpose))

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        detail = result.stdout.strip() or result.stderr.strip().splitlines()[0]
        checks.append(Check("NVIDIA GPU", result.returncode == 0, detail, "training"))
    else:
        checks.append(Check("NVIDIA GPU", False, "nvidia-smi not found", "training"))
    return checks


def environment_as_dicts() -> list[dict[str, object]]:
    return [asdict(check) for check in collect_environment()]


def print_environment() -> bool:
    checks = collect_environment()
    width = max(len(check.name) for check in checks)
    for check in checks:
        mark = "OK" if check.available else "--"
        print(f"[{mark}] {check.name:<{width}}  {check.detail}  ({check.required_for})")
    training_ok = all(
        check.available for check in checks if check.name in {"torch", "gsplat", "NVIDIA GPU"}
    )
    sfm_ok = any(check.available for check in checks if check.name in {"colmap", "pycolmap"})
    print(f"\nSfM ready: {sfm_ok}; training ready: {training_ok}")
    return sfm_ok and training_ok
