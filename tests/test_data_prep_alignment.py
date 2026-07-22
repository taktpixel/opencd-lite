"""Tests for keypoint-based image pair alignment (``opencd_lite.data_prep.alignment``).

Requires OpenCV from the ``dataprep`` extra; the whole module is skipped
when cv2 is not installed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from opencd_lite.data_prep.alignment import (  # noqa: E402
    AlignmentError,
    KeypointAligner,
    align_directories,
)

pytestmark = pytest.mark.dataprep


def _textured_image(seed: int = 0, height: int = 256, width: int = 256) -> np.ndarray:
    """Blurred random rectangles: plenty of stable gradients for keypoints."""
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width), dtype=np.uint8)
    for _ in range(60):
        y = int(rng.integers(0, height - 40))
        x = int(rng.integers(0, width - 40))
        h = int(rng.integers(10, 40))
        w = int(rng.integers(10, 40))
        image[y : y + h, x : x + w] = int(rng.integers(40, 255))
    return cv2.GaussianBlur(image, (5, 5), 0)


class TestKeypointAligner:
    def test_identity_alignment(self):
        image = _textured_image(0)
        result = KeypointAligner(image).align(image)
        assert result.image.shape == image.shape
        assert result.num_matches >= 4
        assert result.matrix.shape == (2, 3)
        np.testing.assert_allclose(result.matrix[:, :2], np.eye(2), atol=0.02)
        np.testing.assert_allclose(result.matrix[:, 2], 0.0, atol=1.0)

    def test_translation_is_recovered(self):
        reference = _textured_image(1)
        target = np.roll(reference, shift=(5, 9), axis=(0, 1))
        result = KeypointAligner(reference).align(target)
        np.testing.assert_allclose(result.matrix[:, :2], np.eye(2), atol=0.05)
        np.testing.assert_allclose(result.matrix[:, 2], [-9.0, -5.0], atol=1.5)
        # Interior pixels line up with the reference after warping.
        aligned = result.image[32:-32, 32:-32].astype(int)
        expected = reference[32:-32, 32:-32].astype(int)
        assert np.abs(aligned - expected).mean() < 8.0

    def test_rigid_transform_has_unit_scale(self):
        reference = _textured_image(2)
        scaled = cv2.resize(reference, None, fx=1.05, fy=1.05, interpolation=cv2.INTER_LINEAR)
        result = KeypointAligner(reference).align(scaled[:256, :256])
        row_norm = float(np.hypot(result.matrix[0, 0], result.matrix[0, 1]))
        assert row_norm == pytest.approx(1.0, abs=1e-6)

    def test_matching_at_reduced_scale(self):
        reference = _textured_image(5)
        target = np.roll(reference, shift=(0, 8), axis=(0, 1))
        result = KeypointAligner(reference, scale=0.5).align(target)
        np.testing.assert_allclose(result.matrix[:, 2], [-8.0, 0.0], atol=2.0)

    def test_orb_method(self):
        image = _textured_image(6)
        result = KeypointAligner(image, method="orb").align(image)
        np.testing.assert_allclose(result.matrix[:, 2], 0.0, atol=2.0)

    def test_color_images_are_supported(self):
        reference = cv2.cvtColor(_textured_image(3), cv2.COLOR_GRAY2BGR)
        result = KeypointAligner(reference).align(reference)
        assert result.image.shape == reference.shape

    def test_featureless_target_raises(self):
        reference = _textured_image(4)
        flat = np.full_like(reference, 128)
        with pytest.raises(AlignmentError):
            KeypointAligner(reference).align(flat)

    def test_featureless_reference_raises(self):
        with pytest.raises(AlignmentError):
            KeypointAligner(np.full((128, 128), 200, dtype=np.uint8))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            KeypointAligner(_textured_image(7), method="surf")


class TestAlignDirectories:
    def test_pairs_aligned_failures_skipped(self, tmp_path):
        ref_dir = tmp_path / "ref"
        tgt_dir = tmp_path / "tgt"
        out_dir = tmp_path / "out"
        debug_dir = tmp_path / "debug"
        ref_dir.mkdir()
        tgt_dir.mkdir()

        for i in (0, 1):
            image = _textured_image(10 + i)
            cv2.imwrite(str(ref_dir / f"pair{i}.png"), image)
            shifted = np.roll(image, shift=(3, 4), axis=(0, 1))
            cv2.imwrite(str(tgt_dir / f"pair{i}.png"), shifted)
        # A pair that cannot be aligned is skipped, not fatal.
        flat = np.full((128, 128), 128, dtype=np.uint8)
        cv2.imwrite(str(ref_dir / "flat.png"), flat)
        cv2.imwrite(str(tgt_dir / "flat.png"), flat)
        # Files without a partner in the other directory are ignored.
        cv2.imwrite(str(tgt_dir / "orphan.png"), _textured_image(99))

        aligned = align_directories(ref_dir, tgt_dir, out_dir, debug_dir=debug_dir)

        assert aligned == ["pair0.png", "pair1.png"]
        assert sorted(p.name for p in out_dir.iterdir()) == ["pair0.png", "pair1.png"]
        payload = json.loads((debug_dir / "pair0_affine.json").read_text())
        assert len(payload["affine_matrix"]) == 6
        assert (debug_dir / "pair0_overlay.png").is_file()
