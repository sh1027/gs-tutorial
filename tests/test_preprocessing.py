from pathlib import Path

import numpy as np
from PIL import Image

from gs_tutorial.preprocessing import difference_hash, hamming_distance, select_images


def _checkerboard(offset: int = 0) -> Image.Image:
    yy, xx = np.indices((128, 128))
    values = ((((xx + offset) // 8 + yy // 8) % 2) * 255).astype(np.uint8)
    return Image.fromarray(np.repeat(values[..., None], 3, axis=2))


def test_difference_hash_is_deterministic() -> None:
    image = _checkerboard()
    assert hamming_distance(difference_hash(image), difference_hash(image.copy())) == 0


def test_select_images_rejects_blur_and_adjacent_duplicate(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    _checkerboard(0).save(source / "001.jpg")
    _checkerboard(0).save(source / "002.jpg")
    _checkerboard(3).save(source / "003.jpg")
    _checkerboard(6).save(source / "004.jpg")
    reports = select_images(
        source,
        output,
        blur_threshold=1.0,
        duplicate_hamming_threshold=0,
    )
    assert reports[0].kept
    assert not reports[1].kept
    assert reports[1].reason == "near duplicate"
    assert sum(report.kept for report in reports) >= 3


def test_select_images_resizes_to_max_width(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    for index, offset in enumerate((0, 3, 6), start=1):
        image = _checkerboard(offset).resize((256, 128))
        image.save(source / f"{index:03}.jpg")

    reports = select_images(
        source,
        output,
        max_width=100,
        blur_threshold=0.0,
        duplicate_hamming_threshold=-1,
    )

    assert all((report.width, report.height) == (100, 50) for report in reports)
    with Image.open(output / "image_00001.jpg") as image:
        assert image.size == (100, 50)
