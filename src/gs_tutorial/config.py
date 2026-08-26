from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class InputConfig:
    source: str = "data/input.mp4"
    kind: str = "video"
    fps: float = 2.0
    max_width: int = 1600
    blur_threshold: float = 25.0
    duplicate_hamming_threshold: int = 4


@dataclass(frozen=True)
class ColmapConfig:
    matcher: str = "sequential"
    camera_model: str = "SIMPLE_RADIAL"
    single_camera: bool = True
    use_gpu: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int = 30_000
    image_downscale: int = 2
    sh_degree: int = 3
    sh_degree_interval: int = 1_000
    l1_weight: float = 0.8
    ssim_weight: float = 0.2
    random_background: bool = False
    checkpoint_every: int = 1_000
    preview_every: int = 100
    log_every: int = 10
    seed: int = 42
    means_lr_final_scale: float = 0.01
    learning_rates: dict[str, float] = field(
        default_factory=lambda: {
            "means": 1.6e-4,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 5e-2,
            "sh0": 2.5e-3,
            "shN": 1.25e-4,
        }
    )


@dataclass(frozen=True)
class PipelineConfig:
    scene_name: str
    input: InputConfig
    colmap: ColmapConfig
    training: TrainingConfig

    @property
    def project_dir(self) -> Path:
        return Path("outputs") / self.scene_name


def _construct(cls: type, values: dict[str, Any] | None):
    values = values or {}
    valid = {field.name for field in cls.__dataclass_fields__.values()}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> PipelineConfig:
    """Load and minimally validate a tutorial YAML configuration."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "scene_name" not in raw:
        raise ValueError("scene_name is required")
    unknown = set(raw) - {"scene_name", "input", "colmap", "training"}
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    cfg = PipelineConfig(
        scene_name=str(raw["scene_name"]),
        input=_construct(InputConfig, raw.get("input")),
        colmap=_construct(ColmapConfig, raw.get("colmap")),
        training=_construct(TrainingConfig, raw.get("training")),
    )
    if Path(cfg.scene_name).name != cfg.scene_name or cfg.scene_name in {"", ".", ".."}:
        raise ValueError("scene_name must be a single safe directory name")
    if cfg.input.kind not in {"video", "images"}:
        raise ValueError("input.kind must be 'video' or 'images'")
    if cfg.input.fps <= 0 or cfg.input.max_width <= 0:
        raise ValueError("input.fps and input.max_width must be positive")
    if cfg.colmap.matcher not in {"sequential", "exhaustive"}:
        raise ValueError("colmap.matcher must be 'sequential' or 'exhaustive'")
    if (
        min(
            cfg.training.max_steps,
            cfg.training.image_downscale,
            cfg.training.sh_degree_interval,
            cfg.training.checkpoint_every,
            cfg.training.preview_every,
            cfg.training.log_every,
        )
        <= 0
    ):
        raise ValueError("training step, interval, and downscale values must be positive")
    if cfg.training.sh_degree < 0:
        raise ValueError("training.sh_degree must be non-negative")
    if cfg.training.l1_weight < 0 or cfg.training.ssim_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if cfg.training.l1_weight + cfg.training.ssim_weight == 0:
        raise ValueError("at least one loss weight must be positive")
    if not 0 < cfg.training.means_lr_final_scale <= 1:
        raise ValueError("training.means_lr_final_scale must be in (0, 1]")
    return cfg
