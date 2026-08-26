from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def intrinsic_matrix(self, scale_x: float = 1.0, scale_y: float | None = None) -> np.ndarray:
        scale_y = scale_x if scale_y is None else scale_y
        if self.model == "PINHOLE":
            fx, fy, cx, cy = self.params[:4]
        elif self.model == "SIMPLE_PINHOLE":
            focal, cx, cy = self.params[:3]
            fx = fy = focal
        else:
            raise ValueError(
                f"Expected an undistorted PINHOLE camera, got {self.model}. "
                "Run COLMAP image_undistorter first."
            )
        return np.array(
            [
                [fx * scale_x, 0.0, cx * scale_x],
                [0.0, fy * scale_y, cy * scale_y],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class RegisteredImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str

    @property
    def world_to_camera(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = qvec_to_rotation(self.qvec)
        matrix[:3, 3] = self.tvec
        return matrix

    @property
    def camera_to_world(self) -> np.ndarray:
        return np.linalg.inv(self.world_to_camera).astype(np.float32)


@dataclass(frozen=True)
class SparseScene:
    cameras: dict[int, Camera]
    images: tuple[RegisteredImage, ...]
    points: np.ndarray
    colors: np.ndarray
    errors: np.ndarray

    @property
    def camera_centers(self) -> np.ndarray:
        return np.stack([image.camera_to_world[:3, 3] for image in self.images])

    @property
    def scene_scale(self) -> float:
        centers = self.camera_centers
        center = np.median(centers, axis=0)
        scale = float(np.median(np.linalg.norm(centers - center, axis=1)))
        return max(scale, 1e-6)


def qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    """COLMAP Hamilton quaternion (w, x, y, z) to a rotation matrix."""
    w, x, y, z = np.asarray(qvec, dtype=np.float64)
    norm = np.linalg.norm([w, x, y, z])
    if norm == 0:
        raise ValueError("zero quaternion")
    w, x, y, z = np.array([w, x, y, z]) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _data_lines(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            yield stripped


def load_colmap_text(model_dir: str | Path) -> SparseScene:
    """Load COLMAP text files after image undistortion."""
    model_dir = Path(model_dir)
    cameras: dict[int, Camera] = {}
    for line in _data_lines(model_dir / "cameras.txt"):
        fields = line.split()
        camera_id, model, width, height = int(fields[0]), fields[1], int(fields[2]), int(fields[3])
        cameras[camera_id] = Camera(
            camera_id, model, width, height, np.asarray(fields[4:], dtype=np.float64)
        )

    lines = [
        line.strip()
        for line in (model_dir / "images.txt").read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]
    while lines and not lines[0]:
        lines.pop(0)
    if len(lines) % 2:
        raise ValueError("Malformed images.txt: every image header needs an observations line")
    images: list[RegisteredImage] = []
    for index in range(0, len(lines), 2):
        fields = lines[index].split()
        if not fields:
            raise ValueError("Malformed images.txt: missing image header")
        images.append(
            RegisteredImage(
                image_id=int(fields[0]),
                qvec=np.asarray(fields[1:5], dtype=np.float64),
                tvec=np.asarray(fields[5:8], dtype=np.float64),
                camera_id=int(fields[8]),
                name=" ".join(fields[9:]),
            )
        )

    points, colors, errors = [], [], []
    for line in _data_lines(model_dir / "points3D.txt"):
        fields = line.split()
        points.append([float(value) for value in fields[1:4]])
        colors.append([int(value) for value in fields[4:7]])
        errors.append(float(fields[7]))
    if not images:
        raise ValueError("The COLMAP model contains no registered images")
    if not points:
        raise ValueError("The COLMAP model contains no sparse points")
    return SparseScene(
        cameras=cameras,
        images=tuple(sorted(images, key=lambda item: item.name)),
        points=np.asarray(points, dtype=np.float32),
        colors=np.asarray(colors, dtype=np.uint8),
        errors=np.asarray(errors, dtype=np.float32),
    )


def load_training_view(
    image_dir: str | Path,
    scene: SparseScene,
    index: int,
    downscale: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return RGB float image, world-to-camera matrix, and scaled intrinsics."""
    record = scene.images[index]
    camera = scene.cameras[record.camera_id]
    with Image.open(Path(image_dir) / record.name) as source:
        image = source.convert("RGB")
        if downscale > 1:
            image = image.resize(
                (image.width // downscale, image.height // downscale), Image.Resampling.LANCZOS
            )
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    scale_x = pixels.shape[1] / camera.width
    scale_y = pixels.shape[0] / camera.height
    return pixels, record.world_to_camera, camera.intrinsic_matrix(float(scale_x), float(scale_y))


def scene_summary(scene: SparseScene) -> dict[str, float | int]:
    return {
        "registered_images": len(scene.images),
        "cameras": len(scene.cameras),
        "sparse_points": len(scene.points),
        "mean_reprojection_error_px": float(scene.errors.mean()),
        "median_reprojection_error_px": float(np.median(scene.errors)),
        "scene_scale": scene.scene_scale,
    }
