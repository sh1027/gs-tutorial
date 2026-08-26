from pathlib import Path

import numpy as np
import pytest

from gs_tutorial.dataset import Camera, RegisteredImage, SparseScene
from gs_tutorial.visualization import (
    _training_snapshot_paths,
    load_gaussian_ply,
    sparse_scene_figure,
)


def _scene() -> SparseScene:
    camera = Camera(1, "PINHOLE", 640, 480, np.array([500, 500, 320, 240]))
    image = RegisteredImage(
        1,
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        1,
        "image.jpg",
    )
    image_2 = RegisteredImage(
        2,
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        1,
        "image_2.jpg",
    )
    return SparseScene(
        {1: camera},
        (image, image_2),
        np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        np.array([[12, 34, 56]], dtype=np.uint8),
        np.array([0.1], dtype=np.float32),
    )


def test_sparse_scene_figure_serializes() -> None:
    pytest.importorskip("plotly")
    figure = sparse_scene_figure(_scene())
    serialized = figure.to_json()
    assert "rgb(12,34,56)" in serialized
    assert "camera path" not in serialized
    assert "camera x axis" in serialized
    assert "camera y axis" in serialized
    assert "camera z axis" in serialized
    frustums = [trace for trace in figure.data if trace.name.startswith("camera frustum ")]
    assert len(frustums) == 2
    assert frustums[0].line.color != frustums[1].line.color
    assert all(trace.showlegend is False for trace in figure.data)
    assert figure.layout.showlegend is False


def test_load_gaussian_ply(tmp_path: Path) -> None:
    pytest.importorskip("plyfile")
    ply = tmp_path / "one.ply"
    ply.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
        "end_header\n"
        "1 2 3 0 0 0 0 0 0 0 1 0 0 0\n",
        encoding="utf-8",
    )
    splats = load_gaussian_ply(ply)
    np.testing.assert_allclose(splats["centers"], [[1, 2, 3]])
    np.testing.assert_allclose(splats["covariances"], np.eye(3)[None])
    np.testing.assert_allclose(splats["rgbs"], [[0.5, 0.5, 0.5]])
    np.testing.assert_allclose(splats["opacities"], [[0.5]])


def test_training_snapshot_paths_are_ordered(tmp_path: Path) -> None:
    for name in (
        "latest.ply",
        "step_000200.ply",
        "final.ply",
        "initial.ply",
        "step_000010.ply",
    ):
        (tmp_path / name).touch()
    assert [path.name for path in _training_snapshot_paths(tmp_path / "final.ply")] == [
        "initial.ply",
        "step_000010.ply",
        "step_000200.ply",
        "final.ply",
    ]
