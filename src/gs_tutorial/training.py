from __future__ import annotations

import json
import math
import random
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .config import TrainingConfig
from .dataset import SparseScene, load_training_view

SH_C0 = 0.28209479177387814
ProgressCallback = Callable[[dict[str, object]], None]


def _require_training_stack():
    try:
        import torch
        from gsplat import DefaultStrategy, export_splats, rasterization
    except ImportError as exc:
        raise RuntimeError(
            "Training requires PyTorch and gsplat. Install the 'gpu' optional dependencies."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("gsplat training requires an NVIDIA CUDA GPU visible to PyTorch")
    return torch, rasterization, DefaultStrategy, export_splats


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def _initial_log_scales(points: np.ndarray, scene_scale: float) -> np.ndarray:
    if len(points) < 2:
        return np.full((len(points), 3), math.log(scene_scale * 0.01), dtype=np.float32)
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("scipy is required to initialize Gaussian scales") from exc
    distances, _ = cKDTree(points).query(points, k=min(4, len(points)))
    if distances.ndim == 1:
        nearest = distances
    else:
        nearest = distances[:, 1:].mean(axis=1)
    nearest = np.clip(nearest, scene_scale * 1e-5, scene_scale * 0.1)
    return np.log(np.repeat(nearest[:, None], 3, axis=1)).astype(np.float32)


def initialize_gaussians(scene: SparseScene, sh_degree: int, device: str = "cuda"):
    """Create the explicit 3DGS parameters from COLMAP sparse points."""
    torch, *_ = _require_training_stack()
    count = len(scene.points)
    basis_count = (sh_degree + 1) ** 2
    sh0 = (scene.colors.astype(np.float32) / 255.0 - 0.5) / SH_C0
    quats = np.zeros((count, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    values = {
        "means": scene.points,
        "scales": _initial_log_scales(scene.points, scene.scene_scale),
        "quats": quats,
        "opacities": np.full(count, _logit(0.1), dtype=np.float32),
        "sh0": sh0[:, None, :],
        "shN": np.zeros((count, basis_count - 1, 3), dtype=np.float32),
    }
    return torch.nn.ParameterDict(
        {
            name: torch.nn.Parameter(torch.as_tensor(value, device=device))
            for name, value in values.items()
        }
    )


def create_optimizers(splats, config: TrainingConfig, scene_scale: float):
    torch, *_ = _require_training_stack()
    rates = dict(config.learning_rates)
    rates["means"] *= scene_scale
    missing = set(splats) - set(rates)
    if missing:
        raise ValueError(f"Missing learning rates for {sorted(missing)}")
    return {
        name: torch.optim.Adam([parameter], lr=rates[name], eps=1e-15)
        for name, parameter in splats.items()
    }


def structural_similarity(prediction, target, window_size: int = 11):
    """Small differentiable SSIM implementation for RGB tensors in NCHW layout."""
    from torch.nn import functional

    padding = window_size // 2
    mu_x = functional.avg_pool2d(prediction, window_size, stride=1, padding=padding)
    mu_y = functional.avg_pool2d(target, window_size, stride=1, padding=padding)
    sigma_x = functional.avg_pool2d(prediction * prediction, window_size, 1, padding) - mu_x**2
    sigma_y = functional.avg_pool2d(target * target, window_size, 1, padding) - mu_y**2
    sigma_xy = functional.avg_pool2d(prediction * target, window_size, 1, padding) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean()


def _export(splats, path: Path, export_splats, format: str = "ply") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    export_splats(
        means=splats["means"],
        scales=splats["scales"],
        quats=splats["quats"],
        opacities=splats["opacities"],
        sh0=splats["sh0"],
        shN=splats["shN"],
        format=format,
        save_to=str(temporary),
    )
    temporary.replace(path)


def train(
    scene: SparseScene,
    image_dir: str | Path,
    output_dir: str | Path,
    config: TrainingConfig,
    callback: ProgressCallback | None = None,
    overwrite: bool = False,
):
    """Train a compact, readable 3DGS model with gsplat's densification strategy."""
    torch, rasterization, DefaultStrategy, export_splats = _require_training_stack()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Training output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} is not empty; use a new project or pass overwrite=True explicitly"
            )
        shutil.rmtree(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    preview_dir = output_dir / "previews"
    ply_dir = output_dir / "point_cloud"
    for directory in (checkpoint_dir, preview_dir, ply_dir):
        directory.mkdir(parents=True, exist_ok=True)

    splats = initialize_gaussians(scene, config.sh_degree)
    optimizers = create_optimizers(splats, config, scene.scene_scale)
    strategy = DefaultStrategy(
        verbose=True,
        refine_start_iter=500,
        refine_stop_iter=min(15_000, config.max_steps),
    )
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene.scene_scale)
    history: list[dict[str, float | int]] = []
    _export(splats, ply_dir / "initial.ply", export_splats)
    _export(splats, ply_dir / "latest.ply", export_splats)

    for step in range(config.max_steps + 1):
        lr_fraction = step / config.max_steps
        means_lr = (
            config.learning_rates["means"]
            * scene.scene_scale
            * config.means_lr_final_scale**lr_fraction
        )
        optimizers["means"].param_groups[0]["lr"] = means_lr
        image_index = random.randrange(len(scene.images))
        pixels_np, viewmat_np, intrinsic_np = load_training_view(
            image_dir, scene, image_index, config.image_downscale
        )
        target = torch.as_tensor(pixels_np, device="cuda")[None]
        viewmat = torch.as_tensor(viewmat_np, device="cuda")[None]
        intrinsic = torch.as_tensor(intrinsic_np, device="cuda")[None]
        height, width = target.shape[1:3]
        background = torch.rand((1, 3), device="cuda") if config.random_background else None
        active_sh_degree = min(step // config.sh_degree_interval, config.sh_degree)
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        rendered, _alpha, info = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors,
            viewmats=viewmat,
            Ks=intrinsic,
            width=width,
            height=height,
            sh_degree=active_sh_degree,
            packed=False,
            backgrounds=background,
        )
        strategy.step_pre_backward(splats, optimizers, strategy_state, step, info)
        l1 = torch.abs(rendered - target).mean()
        ssim = structural_similarity(rendered.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2))
        loss = config.l1_weight * l1 + config.ssim_weight * (1.0 - ssim)
        loss.backward()
        strategy.step_post_backward(splats, optimizers, strategy_state, step, info, packed=False)
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % config.log_every == 0:
            mse = torch.mean((rendered.detach() - target) ** 2).clamp_min(1e-10)
            entry = {
                "step": step,
                "loss": float(loss.detach()),
                "l1": float(l1.detach()),
                "ssim": float(ssim.detach()),
                "psnr": float(-10.0 * torch.log10(mse)),
                "gaussians": len(splats["means"]),
                "opacity_mean": float(torch.sigmoid(splats["opacities"]).detach().mean()),
                "scale_median": float(torch.exp(splats["scales"]).detach().median()),
                "means_lr": means_lr,
            }
            history.append(entry)
            if callback:
                callback(dict(entry))

        if step % config.preview_every == 0:
            comparison = torch.cat([target[0], rendered[0].detach().clamp(0, 1)], dim=1)
            preview = (comparison.cpu().numpy() * 255).astype(np.uint8)
            preview_path = preview_dir / f"step_{step:06d}.jpg"
            Image.fromarray(preview).save(preview_path, quality=92)
            if callback:
                callback({"step": step, "preview": str(preview_path)})

        if step > 0 and step % config.checkpoint_every == 0:
            torch.save(
                {
                    "step": step,
                    "config": asdict(config),
                    "splats": {name: value.detach().cpu() for name, value in splats.items()},
                },
                checkpoint_dir / f"step_{step:06d}.pt",
            )
            _export(splats, ply_dir / f"step_{step:06d}.ply", export_splats)
            _export(splats, ply_dir / "latest.ply", export_splats)

    _export(splats, ply_dir / "final.ply", export_splats, "ply")
    (output_dir / "metrics.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return splats, history
