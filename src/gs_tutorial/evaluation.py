from __future__ import annotations

import html
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import TrainingConfig
from .dataset import SparseScene, load_training_view
from .training import structural_similarity


def _comparison_image(original: np.ndarray, rendered: np.ndarray) -> Image.Image:
    """Build a labeled, side-by-side comparison without changing either image."""
    original_image = Image.fromarray(original)
    rendered_image = Image.fromarray(rendered)
    title_height = 30
    canvas = Image.new(
        "RGB",
        (original_image.width + rendered_image.width, original_image.height + title_height),
        "white",
    )
    canvas.paste(original_image, (0, title_height))
    canvas.paste(rendered_image, (original_image.width, title_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), "ORIGINAL", fill="black")
    draw.text((original_image.width + 10, 8), "RENDER", fill="black")
    return canvas


def _write_gallery(output_dir: Path, rows: list[dict[str, float | str]]) -> None:
    cards = []
    for row in rows:
        name = html.escape(str(row["name"]))
        cards.append(
            f"""<article>
              <a href="comparison/{name}"><img src="comparison/{name}" loading="lazy"></a>
              <p><code>{name}</code><br>PSNR {row["psnr"]:.2f} dB · SSIM {row["ssim"]:.4f}</p>
            </article>"""
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Original / Render comparison</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f4f4f4; }}
main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
article {{ background: white; padding: 10px; border-radius: 8px; box-shadow: 0 1px 5px #bbb; }}
img {{ width: 100%; height: auto; display: block; }} p {{ margin: 8px 2px 2px; }}
</style></head><body><h1>Original / Render comparison</h1><main>{"".join(cards)}</main></body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def render_registered_views(
    scene: SparseScene,
    image_dir: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    config: TrainingConfig,
    *,
    overwrite: bool = False,
    callback: Callable[[dict[str, float | int | str]], None] | None = None,
) -> dict[str, object]:
    """Render every registered COLMAP view from a checkpoint without optimization."""
    try:
        import torch
        from gsplat import rasterization
    except ImportError as exc:
        raise RuntimeError("Rendering requires PyTorch and gsplat") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Rendering requires an NVIDIA CUDA GPU visible to PyTorch")

    checkpoint_path, output_dir = Path(checkpoint_path), Path(output_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Evaluation output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} is not empty; pass overwrite=True explicitly")
        shutil.rmtree(output_dir)
    original_dir = output_dir / "original"
    render_dir = output_dir / "render"
    comparison_dir = output_dir / "comparison"
    for directory in (original_dir, render_dir, comparison_dir):
        directory.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    step = int(checkpoint["step"])
    splats = {name: tensor.to("cuda") for name, tensor in checkpoint["splats"].items()}
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    rows: list[dict[str, float | str]] = []

    with torch.inference_mode():
        for index, record in enumerate(scene.images):
            pixels_np, viewmat_np, intrinsic_np = load_training_view(
                image_dir, scene, index, config.image_downscale
            )
            target = torch.as_tensor(pixels_np, device="cuda")[None]
            rendered, _alpha, _info = rasterization(
                means=splats["means"],
                quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=colors,
                viewmats=torch.as_tensor(viewmat_np, device="cuda")[None],
                Ks=torch.as_tensor(intrinsic_np, device="cuda")[None],
                width=target.shape[2],
                height=target.shape[1],
                sh_degree=config.sh_degree,
                packed=True,
            )
            rendered = rendered.clamp(0, 1)
            l1 = torch.abs(rendered - target).mean()
            mse = torch.mean((rendered - target) ** 2).clamp_min(1e-10)
            ssim = structural_similarity(rendered.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2))
            row: dict[str, float | str] = {
                "name": record.name,
                "l1": float(l1),
                "psnr": float(-10.0 * torch.log10(mse)),
                "ssim": float(ssim),
            }
            rows.append(row)
            original = (target[0].cpu().numpy() * 255).astype(np.uint8)
            render = (rendered[0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(original).save(original_dir / record.name, quality=95)
            Image.fromarray(render).save(render_dir / record.name, quality=95)
            _comparison_image(original, render).save(comparison_dir / record.name, quality=95)
            if callback:
                callback({"index": index + 1, "total": len(scene.images), **row})

    aggregate = {
        key: float(np.mean([float(row[key]) for row in rows])) for key in ("l1", "psnr", "ssim")
    }
    result: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "step": step,
        "views": len(rows),
        "aggregate": aggregate,
        "per_view": rows,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_gallery(output_dir, rows)
    return result
