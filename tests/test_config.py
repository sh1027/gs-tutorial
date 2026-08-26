from pathlib import Path

import pytest

from gs_tutorial.config import load_config


def test_load_config_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("scene_name: test_scene\n", encoding="utf-8")
    config = load_config(path)
    assert config.scene_name == "test_scene"
    assert config.input.kind == "video"
    assert config.training.sh_degree == 3


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("scene_name: test_scene\nsurprise: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown top-level"):
        load_config(path)


def test_unsafe_scene_name_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("scene_name: ../outside\n", encoding="utf-8")
    with pytest.raises(ValueError, match="safe directory"):
        load_config(path)
