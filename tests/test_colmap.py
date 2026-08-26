import sqlite3
from pathlib import Path

import numpy as np

from gs_tutorial.colmap import (
    ColmapPaths,
    colmap_database_summary,
    colmap_stage_status,
    read_feature_keypoints,
    read_strongest_verified_pair,
    select_sparse_model,
)


def _make_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE cameras(camera_id INTEGER)")
        connection.execute("CREATE TABLE images(image_id INTEGER, name TEXT)")
        connection.execute(
            "CREATE TABLE keypoints(image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)"
        )
        connection.execute(
            "CREATE TABLE descriptors(image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)"
        )
        connection.execute(
            "CREATE TABLE matches(pair_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)"
        )
        connection.execute(
            "CREATE TABLE two_view_geometries"
            "(pair_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)"
        )
        connection.executemany("INSERT INTO cameras VALUES (?)", [(1,)])
        connection.executemany("INSERT INTO images VALUES (?, ?)", [(1, "a.jpg"), (2, "b.jpg")])
        keypoints_a = np.arange(40, dtype=np.float32).reshape(10, 4)
        keypoints_b = np.arange(48, dtype=np.float32).reshape(12, 4)
        connection.executemany(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            [(1, 10, 4, keypoints_a.tobytes()), (2, 12, 4, keypoints_b.tobytes())],
        )
        connection.executemany(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            [(1, 10, 128, b""), (2, 12, 128, b"")],
        )
        matches = np.column_stack([np.arange(8), np.arange(8)]).astype(np.uint32)
        verified = matches[:6]
        pair_id = 2_147_483_647 + 2
        connection.executemany(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            [(pair_id, 8, 2, matches.tobytes()), (pair_id + 1, 0, 2, b"")],
        )
        connection.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?)",
            (pair_id, 6, 2, verified.tobytes()),
        )


def test_database_summary_counts_features_and_matches(tmp_path: Path) -> None:
    database = tmp_path / "database.db"
    _make_database(database)

    summary = colmap_database_summary(database)

    assert summary["images"] == 2
    assert summary["keypoints"] == 22
    assert summary["images_with_descriptors"] == 2
    assert summary["matched_pairs"] == 1
    assert summary["matches"] == 8
    assert summary["verified_pairs"] == 1
    assert summary["verified_matches"] == 6


def test_feature_and_verified_match_preview_data(tmp_path: Path) -> None:
    database = tmp_path / "database.db"
    _make_database(database)

    name, keypoints = read_feature_keypoints(database)
    assert name == "b.jpg"
    assert keypoints.shape == (12, 4)

    name_a, name_b, keypoints_a, keypoints_b, matches = read_strongest_verified_pair(database)
    assert (name_a, name_b) == ("a.jpg", "b.jpg")
    assert keypoints_a.shape == (10, 4)
    assert keypoints_b.shape == (12, 4)
    np.testing.assert_array_equal(matches, np.column_stack([np.arange(6), np.arange(6)]))


def test_stage_status_and_model_selection(tmp_path: Path) -> None:
    paths = ColmapPaths.create(tmp_path / "colmap", tmp_path / "images")
    _make_database(paths.database)
    small = paths.sparse_dir / "0"
    large = paths.sparse_dir / "1"
    small.mkdir()
    large.mkdir()
    (small / "images.bin").write_bytes(b"small")
    (large / "images.bin").write_bytes(b"a larger model")

    assert select_sparse_model(paths) == large
    status = colmap_stage_status(paths)
    assert status["features"] is True
    assert status["matching"] is True
    assert status["mapping"] is True
    assert status["undistortion"] is False
    assert status["processed_text"] is False


def test_status_paths_do_not_create_a_workspace(tmp_path: Path) -> None:
    root = tmp_path / "not-created"
    paths = ColmapPaths.from_workspace(root, tmp_path / "images")

    assert colmap_stage_status(paths) == {
        "features": False,
        "matching": False,
        "mapping": False,
        "sparse_text": False,
        "undistortion": False,
        "processed_text": False,
    }
    assert not root.exists()
