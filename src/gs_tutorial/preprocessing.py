from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class FrameReport:
    path: str
    width: int
    height: int
    sharpness: float
    kept: bool
    reason: str


def extract_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    fps: float = 2.0,
    max_width: int = 1600,
    overwrite: bool = False,
) -> list[Path]:
    """Extract consistently oriented JPEG frames with ffmpeg."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for video input")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("frame_*.jpg"))
    if existing and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains extracted frames; pass overwrite=True explicitly"
        )
    output_pattern = output_dir / "frame_%05d.jpg"
    scale = f"scale='min({int(max_width)},iw)':-2"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-vf",
        f"fps={float(fps)},{scale}",
        "-q:v",
        "2",
        str(output_pattern),
    ]
    subprocess.run(command, check=True)
    frames = sorted(output_dir.glob("frame_*.jpg"))
    if len(frames) < 3:
        raise RuntimeError(f"Only {len(frames)} frames were extracted; at least 3 are required")
    return frames


def normalized_image(path: str | Path) -> Image.Image:
    """Apply EXIF orientation and return an RGB image."""
    path = Path(path)
    if path.suffix.lower() in {".heic", ".heif"}:
        try:
            from pillow_heif import register_heif_opener
        except ImportError as exc:
            raise RuntimeError("Install pillow-heif to read HEIC/HEIF smartphone photos") from exc
        register_heif_opener()
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _gray_array(image: Image.Image, size: tuple[int, int] | None = None) -> np.ndarray:
    gray = image.convert("L")
    if size:
        gray = gray.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(gray, dtype=np.float32)


def sharpness_score(image: Image.Image) -> float:
    """Variance of a discrete Laplacian; larger values are usually sharper."""
    gray = _gray_array(image)
    padded = np.pad(gray, 1, mode="edge")
    laplacian = (
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(laplacian.var())


def difference_hash(image: Image.Image, hash_size: int = 8) -> np.ndarray:
    pixels = _gray_array(image, (hash_size + 1, hash_size))
    return pixels[:, 1:] > pixels[:, :-1]


def hamming_distance(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(left != right))


def resize_to_max_width(image: Image.Image, max_width: int | None) -> Image.Image:
    """Downscale an image to ``max_width`` while preserving its aspect ratio."""
    if max_width is None or image.width <= max_width:
        return image
    if max_width <= 0:
        raise ValueError("max_width must be positive")
    height = max(1, round(image.height * max_width / image.width))
    return image.resize((max_width, height), Image.Resampling.LANCZOS)


def select_images(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    max_width: int | None = None,
    blur_threshold: float = 70.0,
    duplicate_hamming_threshold: int = 4,
    copy: bool = True,
) -> list[FrameReport]:
    """Resize and select usable images, preserving file order."""
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    extensions = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
    paths = sorted(path for path in source_dir.iterdir() if path.suffix.lower() in extensions)
    if not paths:
        raise FileNotFoundError(f"No supported images found in {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[FrameReport] = []
    previous_kept_hash: np.ndarray | None = None
    kept_index = 0
    for path in paths:
        try:
            image = normalized_image(path)
        except (OSError, ValueError) as exc:
            reports.append(FrameReport(str(path), 0, 0, 0.0, False, f"decode error: {exc}"))
            continue
        image = resize_to_max_width(image, max_width)
        score = sharpness_score(image)
        current_hash = difference_hash(image)
        reason = "kept"
        kept = True
        if score < blur_threshold:
            kept, reason = False, "blurred"
        elif (
            previous_kept_hash is not None
            and hamming_distance(current_hash, previous_kept_hash) <= duplicate_hamming_threshold
        ):
            kept, reason = False, "near duplicate"
        if kept:
            previous_kept_hash = current_hash
            if copy:
                kept_index += 1
                image.save(output_dir / f"image_{kept_index:05d}.jpg", quality=95)
        reports.append(FrameReport(str(path), image.width, image.height, score, kept, reason))
    if sum(report.kept for report in reports) < 3:
        raise RuntimeError("Fewer than 3 usable images remain; relax filtering or recapture")
    return reports


def reports_as_dicts(reports: list[FrameReport]) -> list[dict[str, object]]:
    return [asdict(report) for report in reports]
