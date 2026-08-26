from pathlib import Path

import numpy as np

from gs_tutorial.evaluation import _comparison_image, _write_gallery


def test_comparison_image_places_views_side_by_side() -> None:
    original = np.zeros((12, 20, 3), dtype=np.uint8)
    rendered = np.full((12, 20, 3), 255, dtype=np.uint8)

    comparison = _comparison_image(original, rendered)

    assert comparison.size == (40, 42)
    pixels = np.asarray(comparison)
    assert np.all(pixels[30:, :20] == 0)
    assert np.all(pixels[30:, 20:] == 255)


def test_gallery_contains_each_view_and_escapes_names(tmp_path: Path) -> None:
    _write_gallery(
        tmp_path,
        [{"name": "view&1.jpg", "l1": 0.01, "psnr": 30.125, "ssim": 0.98765}],
    )

    gallery = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert gallery.count("<article>") == 1
    assert "view&amp;1.jpg" in gallery
    assert "PSNR 30.12 dB" in gallery
    assert "SSIM 0.9877" in gallery
