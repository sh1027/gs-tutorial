from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .dataset import Camera, RegisteredImage, SparseScene

SH_C0 = 0.28209479177387814

_SINGLE_FOCAL_MODELS = {
    "SIMPLE_PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "SIMPLE_RADIAL_FISHEYE",
    "RADIAL_FISHEYE",
}
_DUAL_FOCAL_MODELS = {
    "PINHOLE",
    "OPENCV",
    "OPENCV_FISHEYE",
    "FULL_OPENCV",
    "FOV",
    "THIN_PRISM_FISHEYE",
}


def _focal_lengths(camera: Camera) -> tuple[float, float]:
    if camera.model in _SINGLE_FOCAL_MODELS:
        focal = float(camera.params[0])
        return focal, focal
    if camera.model in _DUAL_FOCAL_MODELS:
        return float(camera.params[0]), float(camera.params[1])
    raise ValueError(f"Unsupported COLMAP camera model for visualization: {camera.model}")


def _frustum_depth(scene: SparseScene) -> float:
    center = np.median(scene.camera_centers, axis=0)
    point_distances = np.linalg.norm(scene.points - center, axis=1)
    finite_distances = point_distances[np.isfinite(point_distances) & (point_distances > 0)]
    point_scale = float(np.median(finite_distances)) if len(finite_distances) else 0.0
    return max(scene.scene_scale * 0.05, point_scale * 0.0125, 1e-4)


def _camera_frustum_vertices(camera: Camera, image: RegisteredImage, depth: float) -> np.ndarray:
    fx, fy = _focal_lengths(camera)
    if camera.model in _SINGLE_FOCAL_MODELS:
        cx, cy = camera.params[1:3]
    else:
        cx, cy = camera.params[2:4]
    pixels = np.array(
        [[0.0, 0.0], [camera.width, 0.0], [camera.width, camera.height], [0.0, camera.height]],
        dtype=np.float32,
    )
    corners_camera = np.column_stack(
        (
            (pixels[:, 0] - cx) * depth / fx,
            (pixels[:, 1] - cy) * depth / fy,
            np.full(4, depth, dtype=np.float32),
        )
    )
    camera_to_world = image.camera_to_world
    corners_world = (camera_to_world[:3, :3] @ corners_camera.T).T + camera_to_world[:3, 3]
    return np.vstack((camera_to_world[:3, 3], corners_world)).astype(np.float32)


def _camera_frustum_lines(camera: Camera, image: RegisteredImage, depth: float) -> np.ndarray:
    edges = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1))
    segments = []
    vertices = _camera_frustum_vertices(camera, image, depth)
    for start, end in edges:
        segments.extend((vertices[start], vertices[end], np.full(3, np.nan)))
    return np.asarray(segments, dtype=np.float32)


def _camera_axis_lines(scene: SparseScene, axis: int) -> np.ndarray:
    segments = []
    axis_length = _frustum_depth(scene) * (2.0 / 3.0)
    for image in scene.images:
        camera_to_world = image.camera_to_world
        center = camera_to_world[:3, 3]
        endpoint = center + camera_to_world[:3, axis] * axis_length
        segments.extend((center, endpoint, np.full(3, np.nan)))
    return np.asarray(segments, dtype=np.float32)


def _average_camera_up(scene: SparseScene) -> np.ndarray:
    local_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    average = np.mean([image.camera_to_world[:3, :3] @ local_up for image in scene.images], axis=0)
    norm = float(np.linalg.norm(average))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (average / norm).astype(np.float32)


def feature_keypoints_figure(
    image_path: str | Path,
    keypoints: np.ndarray,
    *,
    max_keypoints: int = 2_000,
):
    """Overlay a deterministic subset of COLMAP keypoints on an input image."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install the 'notebook' optional dependencies") from exc
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        raise ValueError("keypoints must have shape [N, >=2]")
    indices = np.linspace(0, len(keypoints) - 1, min(len(keypoints), max_keypoints), dtype=int)
    shown = keypoints[indices]
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.imshow(plt.imread(Path(image_path)))
    axis.scatter(shown[:, 0], shown[:, 1], s=5, c="#00e5ff", alpha=0.55, linewidths=0)
    axis.set_title(f"SIFT keypoints: showing {len(shown):,} / {len(keypoints):,}")
    axis.axis("off")
    figure.tight_layout()
    return figure


def verified_matches_figure(
    image_path_a: str | Path,
    image_path_b: str | Path,
    keypoints_a: np.ndarray,
    keypoints_b: np.ndarray,
    matches: np.ndarray,
    *,
    max_matches: int = 120,
):
    """Show a deterministic subset of geometrically verified COLMAP matches."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import ConnectionPatch
    except ImportError as exc:
        raise RuntimeError("Install the 'notebook' optional dependencies") from exc
    if matches.ndim != 2 or matches.shape[1] < 2:
        raise ValueError("matches must have shape [N, >=2]")
    valid = matches[(matches[:, 0] < len(keypoints_a)) & (matches[:, 1] < len(keypoints_b))].astype(
        np.int64
    )
    indices = np.linspace(0, len(valid) - 1, min(len(valid), max_matches), dtype=int)
    shown = valid[indices]
    figure, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(plt.imread(Path(image_path_a)))
    axes[1].imshow(plt.imread(Path(image_path_b)))
    colors = plt.cm.turbo(np.linspace(0, 1, max(len(shown), 1)))
    for (index_a, index_b), color in zip(shown, colors, strict=True):
        point_a, point_b = keypoints_a[index_a, :2], keypoints_b[index_b, :2]
        figure.add_artist(
            ConnectionPatch(
                xyA=point_a,
                xyB=point_b,
                coordsA=axes[0].transData,
                coordsB=axes[1].transData,
                color=color,
                linewidth=0.7,
                alpha=0.65,
            )
        )
    axes[0].set_title(Path(image_path_a).name)
    axes[1].set_title(Path(image_path_b).name)
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"Verified matches: showing {len(shown):,} / {len(valid):,}")
    figure.tight_layout()
    return figure


def sparse_scene_figure(scene: SparseScene, max_points: int = 100_000):
    """Create an interactive Plotly view of a COLMAP reconstruction."""
    try:
        import plotly.graph_objects as go
        from plotly.colors import sample_colorscale
    except ImportError as exc:
        raise RuntimeError("Install the 'notebook' optional dependencies") from exc
    stride = max(1, len(scene.points) // max_points)
    points, colors = scene.points[::stride], scene.colors[::stride]
    point_colors = [f"rgb({red},{green},{blue})" for red, green, blue in colors]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker={"size": 2, "color": point_colors, "symbol": "square"},
            name="sparse points",
            showlegend=False,
        )
    )
    camera_order = np.linspace(0.0, 1.0, len(scene.images)).tolist()
    frustum_colors = sample_colorscale("Turbo", camera_order)
    frustum_depth = _frustum_depth(scene)
    for index, (image, color) in enumerate(zip(scene.images, frustum_colors, strict=True)):
        frustum = _camera_frustum_lines(scene.cameras[image.camera_id], image, frustum_depth)
        figure.add_trace(
            go.Scatter3d(
                x=frustum[:, 0],
                y=frustum[:, 1],
                z=frustum[:, 2],
                mode="lines",
                line={"color": color, "width": 2},
                name=f"camera frustum {index + 1}",
                text=image.name,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )
    for axis, color, name in (
        (0, "#ff0000", "camera x axis"),
        (1, "#00a000", "camera y axis"),
        (2, "#0066ff", "camera z axis"),
    ):
        lines = _camera_axis_lines(scene, axis)
        figure.add_trace(
            go.Scatter3d(
                x=lines[:, 0],
                y=lines[:, 1],
                z=lines[:, 2],
                mode="lines",
                line={"color": color, "width": 3},
                name=name,
                hoverinfo="skip",
                showlegend=False,
            )
        )
    up = _average_camera_up(scene)
    figure.update_layout(
        scene={
            "aspectmode": "data",
            "camera": {"up": {"x": float(up[0]), "y": float(up[1]), "z": float(up[2])}},
            "xaxis": {"backgroundcolor": "#f5f5f5", "gridcolor": "#d8d8d8"},
            "yaxis": {"backgroundcolor": "#f5f5f5", "gridcolor": "#d8d8d8"},
            "zaxis": {"backgroundcolor": "#f5f5f5", "gridcolor": "#d8d8d8"},
        },
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        title="COLMAP sparse reconstruction",
        showlegend=False,
    )
    return figure


def _thumbnail(path: Path, max_width: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if image.width > max_width:
            height = max(1, round(image.height * max_width / image.width))
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        return np.asarray(image)


def _quaternions_to_rotation(q: np.ndarray) -> np.ndarray:
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    rotations = np.empty((len(q), 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotations[:, 0, 1] = 2 * (x * y - w * z)
    rotations[:, 0, 2] = 2 * (x * z + w * y)
    rotations[:, 1, 0] = 2 * (x * y + w * z)
    rotations[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotations[:, 1, 2] = 2 * (y * z - w * x)
    rotations[:, 2, 0] = 2 * (x * z - w * y)
    rotations[:, 2, 1] = 2 * (y * z + w * x)
    rotations[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotations


def load_gaussian_ply(path: str | Path) -> dict[str, np.ndarray]:
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError("Install the 'viewer' optional dependencies") from exc
    vertex = PlyData.read(str(path))["vertex"]
    centers = np.stack([vertex[name] for name in ("x", "y", "z")], axis=1).astype(np.float32)
    log_scales = np.stack([vertex[f"scale_{index}"] for index in range(3)], axis=1)
    quats = np.stack([vertex[f"rot_{index}"] for index in range(4)], axis=1)
    rotations = _quaternions_to_rotation(quats)
    scales_squared = np.exp(log_scales).astype(np.float32) ** 2
    covariances = np.einsum("nij,nj,nkj->nik", rotations, scales_squared, rotations)
    sh0 = np.stack([vertex[f"f_dc_{index}"] for index in range(3)], axis=1)
    rgbs = np.clip(0.5 + SH_C0 * sh0, 0.0, 1.0).astype(np.float32)
    logits = np.asarray(vertex["opacity"], dtype=np.float32)
    opacities = (1.0 / (1.0 + np.exp(-logits)))[:, None]
    return {"centers": centers, "covariances": covariances, "rgbs": rgbs, "opacities": opacities}


def _training_snapshot_paths(path: str | Path) -> list[Path]:
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    snapshots = []
    initial = directory / "initial.ply"
    if initial.is_file():
        snapshots.append(initial)
    snapshots.extend(
        sorted(
            (
                item
                for item in directory.glob("step_*.ply")
                if re.fullmatch(r"step_\d+\.ply", item.name)
            ),
            key=lambda item: int(item.stem.removeprefix("step_")),
        )
    )
    final = directory / "final.ply"
    if final.is_file():
        snapshots.append(final)
    if path.is_file() and path.name != "latest.ply" and path not in snapshots:
        snapshots.append(path)
    return snapshots


def _snapshot_label(path: Path) -> str:
    if path.name == "initial.ply":
        return "initial"
    if path.name == "final.ply":
        return "final"
    match = re.fullmatch(r"step_(\d+)\.ply", path.name)
    return str(int(match.group(1))) if match else path.stem


def _initial_point_size(centers: np.ndarray, scene: SparseScene | None) -> float:
    if scene is not None:
        return max(scene.scene_scale * 0.006, 1e-5)
    extent = np.ptp(centers, axis=0)
    return max(float(np.linalg.norm(extent)) * 0.002, 1e-5)


def _add_training_cameras(server, scene: SparseScene, image_dir: Path, tf):
    root = server.scene.add_frame("/training_cameras", show_axes=False, visible=False)
    frustum_scale = _frustum_depth(scene)
    frustums = []
    for record in scene.images:
        camera = scene.cameras[record.camera_id]
        _, fy = _focal_lengths(camera)
        camera_to_world = record.camera_to_world
        frame = server.scene.add_frame(
            f"/training_cameras/frame_{record.image_id}",
            wxyz=tf.SO3.from_matrix(camera_to_world[:3, :3]).wxyz,
            position=camera_to_world[:3, 3],
            axes_length=frustum_scale * (2.0 / 3.0),
            axes_radius=frustum_scale / 30.0,
        )
        image = _thumbnail(image_dir / record.name, 320)
        frustum = server.scene.add_camera_frustum(
            f"/training_cameras/frame_{record.image_id}/frustum",
            fov=float(2.0 * np.arctan2(camera.height / 2.0, fy)),
            aspect=camera.width / camera.height,
            scale=frustum_scale,
            color=(20, 20, 20),
            image=image,
        )
        frustums.append((frustum, image))

        @frustum.on_click
        def _(_, frame=frame) -> None:
            for client in server.get_clients().values():
                client.camera.wxyz = frame.wxyz
                client.camera.position = frame.position

    return root, frustums


class GaussianViewer:
    """Handle for a non-blocking Gaussian Viser server."""

    def __init__(self, server, stop_event: threading.Event, watch_thread: threading.Thread | None):
        self.server = server
        self._stop_event = stop_event
        self._watch_thread = watch_thread

    def stop(self) -> None:
        """Stop the watcher and release the server port."""
        self._stop_event.set()
        if self._watch_thread is not None and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=2.0)
        self.server.stop()


def start_gaussian_viewer(
    path: str | Path,
    *,
    port: int = 8080,
    watch: bool = False,
    poll_seconds: float = 1.0,
    scene: SparseScene | None = None,
    image_dir: str | Path | None = None,
) -> GaussianViewer:
    """Start a non-blocking Gaussian viewer with training and camera controls."""
    try:
        import viser
        import viser.transforms as tf
    except ImportError as exc:
        raise RuntimeError("Install the 'viewer' optional dependencies") from exc
    path = Path(path)
    if path.is_dir():
        candidates = _training_snapshot_paths(path)
        if candidates:
            path = candidates[-1]
        elif watch:
            path = path / "latest.ply"
        else:
            raise FileNotFoundError(f"No training PLY snapshots found in {path}")
    if not watch and not path.is_file():
        raise FileNotFoundError(path)

    server = viser.ViserServer(port=port)
    server.gui.configure_theme(titlebar_content=None)
    server.gui.main_panel.dock_right()
    if scene is not None:
        server.scene.set_up_direction(_average_camera_up(scene))

    info = server.gui.add_markdown("")
    state = {"gaussians": None, "initial_points": None}

    def update_info(selected: Path, count: int) -> None:
        info.content = f"**Snapshot:** `{selected.name}`  \n**Gaussians:** {count:,}"

    gui_gaussians = server.gui.add_checkbox("Show Gaussians", initial_value=True)
    gui_initial_points = server.gui.add_checkbox("Show initial point cloud", initial_value=False)

    def load_gaussians(selected: Path) -> None:
        data = load_gaussian_ply(selected)
        handle = state["gaussians"]
        if handle is None:
            handle = server.scene.add_gaussian_splats(
                "/gaussians", visible=gui_gaussians.value, **data
            )
            state["gaussians"] = handle
        else:
            handle.set_gaussians(**data)
        update_info(selected, len(data["centers"]))

    def load_initial_points() -> None:
        if state["initial_points"] is not None:
            return
        initial_path = path.parent / "initial.ply"
        if not initial_path.is_file():
            return
        initial = load_gaussian_ply(initial_path)
        state["initial_points"] = server.scene.add_point_cloud(
            "/initial_point_cloud",
            points=initial["centers"],
            colors=initial["rgbs"],
            point_size=_initial_point_size(initial["centers"], scene),
            point_shape="circle",
            visible=gui_initial_points.value,
        )

    if path.is_file():
        load_gaussians(path)
    else:
        info.content = f"Waiting for training snapshot: `{path.name}`"
    load_initial_points()

    @gui_gaussians.on_update
    def _(_) -> None:
        if state["gaussians"] is not None:
            state["gaussians"].visible = gui_gaussians.value

    @gui_initial_points.on_update
    def _(_) -> None:
        if state["initial_points"] is not None:
            state["initial_points"].visible = gui_initial_points.value

    if not watch:
        snapshots = _training_snapshot_paths(path)
        if len(snapshots) > 1:
            initial_index = snapshots.index(path) if path in snapshots else len(snapshots) - 1
            gui_step = server.gui.add_slider(
                "Training step",
                min=0,
                max=len(snapshots) - 1,
                step=1,
                initial_value=initial_index,
                marks=tuple((index, _snapshot_label(item)) for index, item in enumerate(snapshots)),
            )

            @gui_step.on_update
            def _(_) -> None:
                selected = snapshots[int(gui_step.value)]
                load_gaussians(selected)

    if scene is not None:
        camera_root, camera_frustums = _add_training_cameras(
            server,
            scene,
            Path(image_dir) if image_dir is not None else Path(),
            tf,
        )
        gui_cameras = server.gui.add_checkbox("Show cameras", initial_value=False)
        gui_camera_images = server.gui.add_checkbox("Show camera images", initial_value=True)

        @gui_cameras.on_update
        def _(_) -> None:
            camera_root.visible = gui_cameras.value

        @gui_camera_images.on_update
        def _(_) -> None:
            for frustum, image in camera_frustums:
                frustum.image = image if gui_camera_images.value else None

    stop_event = threading.Event()
    watch_thread = None
    if watch:
        initial_modified = path.stat().st_mtime_ns if path.is_file() else None

        def watch_snapshots() -> None:
            modified = initial_modified
            while not stop_event.wait(poll_seconds):
                try:
                    load_initial_points()
                    if not path.is_file():
                        continue
                    current = path.stat().st_mtime_ns
                    if current == modified:
                        continue
                    load_gaussians(path)
                    modified = current
                    print(f"Reloaded Gaussian snapshot: {path}")
                except (OSError, ValueError) as exc:
                    print(f"Waiting for a complete Gaussian snapshot: {exc}")

        watch_thread = threading.Thread(
            target=watch_snapshots,
            name="gs-tutorial-viewer-watch",
            daemon=True,
        )
        watch_thread.start()

    print(f"Open the Viser URL above. Serving {path}")
    return GaussianViewer(server, stop_event, watch_thread)


def serve_ply(
    path: str | Path,
    *,
    port: int = 8080,
    watch: bool = False,
    poll_seconds: float = 1.0,
    scene: SparseScene | None = None,
    image_dir: str | Path | None = None,
) -> None:
    """Serve Gaussian snapshots until interrupted from the command line."""
    viewer = start_gaussian_viewer(
        path,
        port=port,
        watch=watch,
        poll_seconds=poll_seconds,
        scene=scene,
        image_dir=image_dir,
    )
    try:
        while True:
            time.sleep(3600.0)
    except KeyboardInterrupt:
        print("Viewer stopped")
    finally:
        viewer.stop()
