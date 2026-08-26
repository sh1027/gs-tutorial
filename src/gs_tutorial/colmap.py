from __future__ import annotations

import importlib.util
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

Backend = Literal["auto", "cli", "pycolmap"]
Matcher = Literal["sequential", "exhaustive"]


@dataclass(frozen=True)
class ColmapPaths:
    root: Path
    image_dir: Path
    database: Path
    sparse_dir: Path
    sparse_text_dir: Path
    processed_dir: Path
    text_model_dir: Path

    @classmethod
    def from_workspace(cls, root: str | Path, image_dir: str | Path) -> ColmapPaths:
        """Describe workspace paths without touching the filesystem."""
        root = Path(root)
        return cls(
            root=root,
            image_dir=Path(image_dir),
            database=root / "database.db",
            sparse_dir=root / "sparse",
            sparse_text_dir=root / "sparse_txt",
            processed_dir=root / "processed",
            text_model_dir=root / "processed" / "sparse_txt",
        )

    @classmethod
    def create(cls, root: str | Path, image_dir: str | Path) -> ColmapPaths:
        paths = cls.from_workspace(root, image_dir)
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.sparse_dir.mkdir(parents=True, exist_ok=True)
        return paths


def resolve_backend(backend: Backend = "auto") -> Literal["cli", "pycolmap"]:
    """Resolve one backend so every independently run stage uses the same implementation."""
    if backend == "auto":
        if importlib.util.find_spec("pycolmap") is not None:
            return "pycolmap"
        if shutil.which("colmap"):
            return "cli"
        raise RuntimeError("Neither PyCOLMAP nor the COLMAP executable is installed")
    if backend == "cli" and shutil.which("colmap") is None:
        raise RuntimeError("COLMAP executable not found; use backend='pycolmap' or install COLMAP")
    if backend == "pycolmap" and importlib.util.find_spec("pycolmap") is None:
        raise RuntimeError("Install the 'reconstruction' optional dependencies")
    return backend


def _run(command: list[str], verbose: bool = True) -> None:
    if verbose:
        print("$", " ".join(command))
    subprocess.run(command, check=True)


def _colmap_executable() -> str:
    executable = shutil.which("colmap")
    if executable is None:
        raise RuntimeError("COLMAP executable not found")
    return executable


def _require_database(paths: ColmapPaths) -> None:
    if not paths.database.is_file():
        raise FileNotFoundError(f"Run feature extraction first: {paths.database}")


def _require_empty_directory(path: Path, stage: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"{stage} output is not empty: {path}")


def _model_score(path: Path) -> int:
    for name in ("images.bin", "images.txt"):
        candidate = path / name
        if candidate.is_file():
            return candidate.stat().st_size
    return -1


def _largest_model(sparse_dir: Path) -> Path:
    candidates = [path for path in sparse_dir.iterdir() if path.is_dir()]
    candidates = [path for path in candidates if _model_score(path) >= 0]
    if not candidates:
        raise RuntimeError(f"COLMAP produced no sparse model under {sparse_dir}")
    return max(candidates, key=_model_score)


def select_sparse_model(paths: ColmapPaths) -> Path:
    """Return the largest available sparse model from the mapping stage."""
    return _largest_model(paths.sparse_dir)


def _pycolmap_device(use_gpu: bool):
    import pycolmap

    has_cuda = bool(getattr(pycolmap, "has_cuda", False))
    if use_gpu and not has_cuda:
        print("PyCOLMAP was installed without CUDA; falling back to CPU feature processing")
    return pycolmap.Device.cuda if use_gpu and has_cuda else pycolmap.Device.cpu


def extract_features(
    paths: ColmapPaths,
    *,
    backend: Backend = "auto",
    camera_model: str = "SIMPLE_RADIAL",
    single_camera: bool = True,
    use_gpu: bool = True,
    verbose: bool = True,
) -> None:
    """Stage 1: detect SIFT keypoints and descriptors and create database.db."""
    if paths.database.exists():
        raise FileExistsError(f"Feature database already exists: {paths.database}")
    if not paths.image_dir.is_dir():
        raise NotADirectoryError(paths.image_dir)
    selected = resolve_backend(backend)
    if verbose:
        print("[1/6] Extracting SIFT features")
    if selected == "cli":
        _run(
            [
                _colmap_executable(),
                "feature_extractor",
                "--database_path",
                str(paths.database),
                "--image_path",
                str(paths.image_dir),
                "--ImageReader.camera_model",
                camera_model,
                "--ImageReader.single_camera",
                "1" if single_camera else "0",
                "--SiftExtraction.use_gpu",
                "1" if use_gpu else "0",
            ],
            verbose,
        )
        return

    import pycolmap

    camera_mode = pycolmap.CameraMode.SINGLE if single_camera else pycolmap.CameraMode.AUTO
    pycolmap.extract_features(
        database_path=paths.database,
        image_path=paths.image_dir,
        camera_mode=camera_mode,
        reader_options=pycolmap.ImageReaderOptions(camera_model=camera_model),
        device=_pycolmap_device(use_gpu),
    )


def match_features(
    paths: ColmapPaths,
    *,
    backend: Backend = "auto",
    matcher: Matcher = "sequential",
    use_gpu: bool = True,
    verbose: bool = True,
) -> None:
    """Stage 2: create and geometrically verify feature correspondences."""
    _require_database(paths)
    selected = resolve_backend(backend)
    if verbose:
        print(f"[2/6] Running {matcher} matching")
    if selected == "cli":
        matcher_command = "sequential_matcher" if matcher == "sequential" else "exhaustive_matcher"
        _run(
            [
                _colmap_executable(),
                matcher_command,
                "--database_path",
                str(paths.database),
                "--SiftMatching.use_gpu",
                "1" if use_gpu else "0",
            ],
            verbose,
        )
        return

    import pycolmap

    match = pycolmap.match_sequential if matcher == "sequential" else pycolmap.match_exhaustive
    match(database_path=paths.database, device=_pycolmap_device(use_gpu))


def map_sparse(
    paths: ColmapPaths,
    *,
    backend: Backend = "auto",
    verbose: bool = True,
) -> Path:
    """Stage 3: estimate camera poses and triangulate a sparse point cloud."""
    _require_database(paths)
    _require_empty_directory(paths.sparse_dir, "Sparse mapping")
    selected = resolve_backend(backend)
    if verbose:
        print("[3/6] Incremental mapping")
    if selected == "cli":
        _run(
            [
                _colmap_executable(),
                "mapper",
                "--database_path",
                str(paths.database),
                "--image_path",
                str(paths.image_dir),
                "--output_path",
                str(paths.sparse_dir),
            ],
            verbose,
        )
        return select_sparse_model(paths)

    import pycolmap

    reconstructions = pycolmap.incremental_mapping(
        database_path=paths.database,
        image_path=paths.image_dir,
        output_path=paths.sparse_dir,
    )
    if not reconstructions:
        raise RuntimeError("PyCOLMAP produced no sparse reconstruction")
    model_id = max(reconstructions, key=lambda key: reconstructions[key].num_reg_images())
    return paths.sparse_dir / str(model_id)


def export_sparse_text(
    paths: ColmapPaths,
    *,
    backend: Backend = "auto",
    model_dir: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    """Stage 4: export the mapped, still-distorted sparse model for inspection."""
    model_dir = Path(model_dir) if model_dir else select_sparse_model(paths)
    _require_empty_directory(paths.sparse_text_dir, "Sparse text export")
    paths.sparse_text_dir.mkdir(parents=True, exist_ok=True)
    selected = resolve_backend(backend)
    if verbose:
        print("[4/6] Exporting the mapped sparse model for inspection")
    if selected == "cli":
        _run(
            [
                _colmap_executable(),
                "model_converter",
                "--input_path",
                str(model_dir),
                "--output_path",
                str(paths.sparse_text_dir),
                "--output_type",
                "TXT",
            ],
            verbose,
        )
    else:
        import pycolmap

        pycolmap.Reconstruction(model_dir).write_text(paths.sparse_text_dir)
    return paths.sparse_text_dir


def undistort_images(
    paths: ColmapPaths,
    *,
    backend: Backend = "auto",
    model_dir: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    """Stage 5: create pinhole images and a matching sparse model for gsplat."""
    model_dir = Path(model_dir) if model_dir else select_sparse_model(paths)
    _require_empty_directory(paths.processed_dir, "Undistortion")
    selected = resolve_backend(backend)
    if verbose:
        print("[5/6] Undistorting registered images")
    if selected == "cli":
        _run(
            [
                _colmap_executable(),
                "image_undistorter",
                "--image_path",
                str(paths.image_dir),
                "--input_path",
                str(model_dir),
                "--output_path",
                str(paths.processed_dir),
                "--output_type",
                "COLMAP",
            ],
            verbose,
        )
    else:
        import pycolmap

        pycolmap.undistort_images(
            output_path=paths.processed_dir,
            input_path=model_dir,
            image_path=paths.image_dir,
        )
    return paths.processed_dir


def export_processed_text(
    paths: ColmapPaths,
    *,
    backend: Backend = "auto",
    verbose: bool = True,
) -> Path:
    """Stage 6: export the undistorted model to the text format used by the tutorial."""
    processed_sparse = paths.processed_dir / "sparse"
    if not processed_sparse.is_dir():
        raise FileNotFoundError(f"Run image undistortion first: {processed_sparse}")
    _require_empty_directory(paths.text_model_dir, "Processed text export")
    paths.text_model_dir.mkdir(parents=True, exist_ok=True)
    selected = resolve_backend(backend)
    if verbose:
        print("[6/6] Exporting the undistorted text model")
    if selected == "cli":
        _run(
            [
                _colmap_executable(),
                "model_converter",
                "--input_path",
                str(processed_sparse),
                "--output_path",
                str(paths.text_model_dir),
                "--output_type",
                "TXT",
            ],
            verbose,
        )
    else:
        import pycolmap

        pycolmap.Reconstruction(processed_sparse).write_text(paths.text_model_dir)
    return paths.text_model_dir


def _database_value(
    connection: sqlite3.Connection,
    table: str,
    expression: str,
    where: str | None = None,
) -> int:
    tables = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    if table not in tables:
        return 0
    query = f"SELECT {expression} FROM {table}"
    if where:
        query += f" WHERE {where}"
    value = connection.execute(query).fetchone()[0]
    return int(value or 0)


def colmap_database_summary(database: str | Path) -> dict[str, int]:
    """Read feature and match counts directly from a COLMAP SQLite database."""
    database = Path(database)
    if not database.is_file():
        raise FileNotFoundError(database)
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        return {
            "cameras": _database_value(connection, "cameras", "COUNT(*)"),
            "images": _database_value(connection, "images", "COUNT(*)"),
            "images_with_keypoints": _database_value(
                connection, "keypoints", "COUNT(*)", "rows > 0"
            ),
            "keypoints": _database_value(connection, "keypoints", "SUM(rows)"),
            "images_with_descriptors": _database_value(
                connection, "descriptors", "COUNT(*)", "rows > 0"
            ),
            "descriptors": _database_value(connection, "descriptors", "SUM(rows)"),
            "matched_pairs": _database_value(connection, "matches", "COUNT(*)", "rows > 0"),
            "matches": _database_value(connection, "matches", "SUM(rows)"),
            "verified_pairs": _database_value(
                connection, "two_view_geometries", "COUNT(*)", "rows > 0"
            ),
            "verified_matches": _database_value(connection, "two_view_geometries", "SUM(rows)"),
        }


def read_feature_keypoints(
    database: str | Path, image_name: str | None = None
) -> tuple[str, np.ndarray]:
    """Read keypoints for one image, choosing the richest image when no name is given."""
    database = Path(database)
    if not database.is_file():
        raise FileNotFoundError(database)
    query = (
        "SELECT images.name, keypoints.rows, keypoints.cols, keypoints.data "
        "FROM keypoints JOIN images USING(image_id) "
    )
    parameters: tuple[str, ...] = ()
    if image_name is None:
        query += "WHERE keypoints.rows > 0 ORDER BY keypoints.rows DESC LIMIT 1"
    else:
        query += "WHERE images.name = ? AND keypoints.rows > 0 LIMIT 1"
        parameters = (image_name,)
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        row = connection.execute(query, parameters).fetchone()
    if row is None:
        requested = image_name or "any image"
        raise ValueError(f"No keypoints found for {requested}")
    name, rows, columns, data = row
    keypoints = np.frombuffer(data, dtype=np.float32).reshape(int(rows), int(columns)).copy()
    return str(name), keypoints


def read_strongest_verified_pair(
    database: str | Path,
) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray]:
    """Read the image pair with the most geometrically verified correspondences."""
    database = Path(database)
    if not database.is_file():
        raise FileNotFoundError(database)
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        pair = connection.execute(
            "SELECT pair_id, rows, cols, data FROM two_view_geometries "
            "WHERE rows > 0 ORDER BY rows DESC LIMIT 1"
        ).fetchone()
        if pair is None:
            raise ValueError("No geometrically verified image pair was found")
        pair_id, rows, columns, data = pair
        max_image_id = 2_147_483_647
        image_id_a = int(pair_id) // max_image_id
        image_id_b = int(pair_id) % max_image_id

        def image_and_keypoints(image_id: int) -> tuple[str, np.ndarray]:
            result = connection.execute(
                "SELECT images.name, keypoints.rows, keypoints.cols, keypoints.data "
                "FROM keypoints JOIN images USING(image_id) WHERE images.image_id = ?",
                (image_id,),
            ).fetchone()
            if result is None:
                raise ValueError(f"Missing keypoints for image id {image_id}")
            name, count, dimensions, keypoint_data = result
            keypoints = np.frombuffer(keypoint_data, dtype=np.float32).reshape(
                int(count), int(dimensions)
            )
            return str(name), keypoints.copy()

        name_a, keypoints_a = image_and_keypoints(image_id_a)
        name_b, keypoints_b = image_and_keypoints(image_id_b)
    matches = np.frombuffer(data, dtype=np.uint32).reshape(int(rows), int(columns)).copy()
    return name_a, name_b, keypoints_a, keypoints_b, matches[:, :2]


def sparse_model_summary(model_dir: str | Path) -> dict[str, float | int]:
    """Summarize a binary or text sparse model without converting it first."""
    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError("Install the 'reconstruction' optional dependencies") from exc
    reconstruction = pycolmap.Reconstruction(Path(model_dir))
    errors = [float(point.error) for point in reconstruction.points3D.values()]
    return {
        "registered_images": int(reconstruction.num_reg_images()),
        "cameras": len(reconstruction.cameras),
        "sparse_points": int(reconstruction.num_points3D()),
        "mean_reprojection_error_px": sum(errors) / len(errors) if errors else 0.0,
    }


def colmap_stage_status(paths: ColmapPaths) -> dict[str, bool]:
    """Report which independently runnable COLMAP stages have usable outputs."""
    database = colmap_database_summary(paths.database) if paths.database.is_file() else {}
    mapped = any(_model_score(path) >= 0 for path in paths.sparse_dir.glob("*") if path.is_dir())
    return {
        "features": bool(database.get("images_with_keypoints", 0)),
        "matching": bool(database.get("verified_pairs", 0)),
        "mapping": mapped,
        "sparse_text": all(
            (paths.sparse_text_dir / name).is_file()
            for name in ("cameras.txt", "images.txt", "points3D.txt")
        ),
        "undistortion": (paths.processed_dir / "images").is_dir()
        and (paths.processed_dir / "sparse").is_dir(),
        "processed_text": all(
            (paths.text_model_dir / name).is_file()
            for name in ("cameras.txt", "images.txt", "points3D.txt")
        ),
    }


def _run_stages(
    image_dir: str | Path,
    workspace: str | Path,
    *,
    backend: Literal["cli", "pycolmap"],
    matcher: Matcher,
    camera_model: str,
    single_camera: bool,
    use_gpu: bool,
    verbose: bool,
) -> ColmapPaths:
    paths = ColmapPaths.create(workspace, image_dir)
    extract_features(
        paths,
        backend=backend,
        camera_model=camera_model,
        single_camera=single_camera,
        use_gpu=use_gpu,
        verbose=verbose,
    )
    match_features(paths, backend=backend, matcher=matcher, use_gpu=use_gpu, verbose=verbose)
    model_dir = map_sparse(paths, backend=backend, verbose=verbose)
    undistort_images(paths, backend=backend, model_dir=model_dir, verbose=verbose)
    export_processed_text(paths, backend=backend, verbose=verbose)
    return paths


def run_colmap_cli(
    image_dir: str | Path,
    workspace: str | Path,
    *,
    matcher: Matcher = "sequential",
    camera_model: str = "SIMPLE_RADIAL",
    single_camera: bool = True,
    use_gpu: bool = True,
    verbose: bool = True,
) -> ColmapPaths:
    """Run all COLMAP CLI stages and prepare an undistorted gsplat dataset."""
    resolve_backend("cli")
    return _run_stages(
        image_dir,
        workspace,
        backend="cli",
        matcher=matcher,
        camera_model=camera_model,
        single_camera=single_camera,
        use_gpu=use_gpu,
        verbose=verbose,
    )


def run_pycolmap(
    image_dir: str | Path,
    workspace: str | Path,
    *,
    matcher: Matcher = "sequential",
    camera_model: str = "SIMPLE_RADIAL",
    single_camera: bool = True,
    use_gpu: bool = True,
    verbose: bool = True,
) -> ColmapPaths:
    """Run all PyCOLMAP stages and prepare an undistorted gsplat dataset."""
    resolve_backend("pycolmap")
    return _run_stages(
        image_dir,
        workspace,
        backend="pycolmap",
        matcher=matcher,
        camera_model=camera_model,
        single_camera=single_camera,
        use_gpu=use_gpu,
        verbose=verbose,
    )


def run_reconstruction(
    image_dir: str | Path,
    workspace: str | Path,
    *,
    backend: Backend = "auto",
    **kwargs,
) -> ColmapPaths:
    """Run the complete pipeline; use the stage functions for interactive execution."""
    selected = resolve_backend(backend)
    if selected == "cli":
        return run_colmap_cli(image_dir, workspace, **kwargs)
    return run_pycolmap(image_dir, workspace, **kwargs)
