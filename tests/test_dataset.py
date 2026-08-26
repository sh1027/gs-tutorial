from pathlib import Path

import numpy as np

from gs_tutorial.dataset import Camera, load_colmap_text, qvec_to_rotation, scene_summary


def test_identity_quaternion() -> None:
    np.testing.assert_allclose(qvec_to_rotation(np.array([1.0, 0.0, 0.0, 0.0])), np.eye(3))


def test_intrinsics_support_independent_rounding_scales() -> None:
    camera = Camera(1, "PINHOLE", 1579, 888, np.array([1400, 1400, 789.5, 444]))
    intrinsic = camera.intrinsic_matrix(394 / 1579, 222 / 888)
    np.testing.assert_allclose(intrinsic[0, 0], 1400 * 394 / 1579)
    np.testing.assert_allclose(intrinsic[1, 1], 350)


def test_load_colmap_text(tmp_path: Path) -> None:
    (tmp_path / "cameras.txt").write_text(
        "# camera list\n1 PINHOLE 640 480 500 510 320 240\n", encoding="utf-8"
    )
    (tmp_path / "images.txt").write_text(
        "# image list\n"
        "1 1 0 0 0 0 0 0 1 image_00001.jpg\n"
        "10 20 1 30 40 2\n"
        "2 1 0 0 0 1 0 0 1 image_00002.jpg\n"
        "11 21 1\n",
        encoding="utf-8",
    )
    (tmp_path / "points3D.txt").write_text(
        "# points\n1 0 0 2 255 0 0 0.25 1 0 2 0\n2 1 0 2 0 255 0 0.75 1 1\n",
        encoding="utf-8",
    )
    scene = load_colmap_text(tmp_path)
    assert len(scene.images) == 2
    assert scene.cameras[1].model == "PINHOLE"
    np.testing.assert_allclose(scene.images[0].world_to_camera, np.eye(4))
    np.testing.assert_allclose(scene.images[1].camera_to_world[:3, 3], [-1, 0, 0])
    assert scene_summary(scene)["sparse_points"] == 2


def test_image_with_no_observations_is_parsed(tmp_path: Path) -> None:
    (tmp_path / "cameras.txt").write_text(
        "1 SIMPLE_PINHOLE 640 480 500 320 240\n", encoding="utf-8"
    )
    (tmp_path / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 empty.jpg\n\n2 1 0 0 0 1 0 0 1 observed.jpg\n10 20 1\n",
        encoding="utf-8",
    )
    (tmp_path / "points3D.txt").write_text("1 0 0 2 255 255 255 0.5 2 0\n", encoding="utf-8")
    scene = load_colmap_text(tmp_path)
    assert [image.name for image in scene.images] == ["empty.jpg", "observed.jpg"]
