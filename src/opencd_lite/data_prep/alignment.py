"""Keypoint-based alignment of image pairs (requires OpenCV / ``dataprep``).

This is the only data_prep module that imports cv2 at module top. It is a
heavily slimmed port of the ari-data-preprocess
``ImageAlignerKeypointMatcher`` / ``KeypointMatcher``: the disabled
GrabPreprocessor mask, the dead FFT stripe removal and edge preprocessing,
the per-step performance logging and the match-visualization debug image are
all dropped. What is kept: CLAHE + brightness preprocessing, a brightness
based keypoint mask, the Lowe ratio test, a RANSAC partial-affine estimate
with an optional rigid (scale-removal) step, matching at a reduced scale with
matrix rescaling, and overlay / affine-json debug output.

Two ari bugs are fixed here:

* ari always used ``NORM_L2`` in the matcher, which is wrong for the binary
  descriptors produced by ORB/AKAZE; those methods now use ``NORM_HAMMING``.
* ari's directory command paired reference/target files by sort order, which
  silently mispaired when the directories differed; :func:`align_directories`
  pairs strictly by identical file name instead.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2 as _cv2
import numpy as np

# OpenCV ships incomplete, version-dependent type stubs (e.g. top-level
# ``SIFT_create`` and the ``imread`` return type differ between builds), so
# alias the module to ``Any`` to keep static typing independent of the
# installed OpenCV version.
cv2: Any = _cv2

__all__ = ["AlignmentError", "AlignmentResult", "KeypointAligner", "align_directories"]

logger = logging.getLogger(__name__)


class AlignmentError(RuntimeError):
    """Raised when a pair cannot be aligned (too few matches, no transform)."""


@dataclass(frozen=True)
class AlignmentResult:
    """Result of aligning a target image onto a reference frame."""

    image: np.ndarray
    """Aligned target warped into the reference frame, sized
    ``(reference_H, reference_W)`` with the target's channel count."""
    matrix: np.ndarray
    """``(2, 3)`` affine mapping target coords to reference coords, corrected
    back to full resolution."""
    num_matches: int
    """Number of good matches surviving the ratio test."""


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Return a single-channel view of ``image`` (BGR->gray if 3-channel)."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


class KeypointAligner:
    """Align target images onto a fixed reference via keypoint matching.

    The reference features are computed once at construction time.

    Args:
        reference: Reference image, uint8 ``(H, W)`` or ``(H, W, 3)``.
        method: Detector, one of ``'sift'``, ``'orb'`` or ``'akaze'``.
        scale: Downscale factor applied for matching only (matrices are
            corrected back to full resolution).
        clahe: Whether to apply CLAHE during preprocessing.
        brightness_offset: ``convertScaleAbs`` beta (alpha fixed at ``1.0``).
        brightness_mask_threshold: Drop target keypoints whose preprocessed
            gray value exceeds ``threshold * 255`` (excludes specular
            highlights); ``None`` disables the mask.
        ratio_threshold: Lowe ratio-test threshold.
        ransac_reproj_threshold: RANSAC reprojection threshold.
        ransac_max_iters: RANSAC maximum iterations.
        ransac_confidence: RANSAC confidence.
        rigid: Remove scale from the estimated transform (rigid alignment).

    Raises:
        ValueError: If ``method`` is not one of the supported detectors.
        AlignmentError: If the reference yields no descriptors.
    """

    def __init__(
        self,
        reference: np.ndarray,
        *,
        method: str = "sift",
        scale: float = 1.0,
        clahe: bool = True,
        brightness_offset: float = -30.0,
        brightness_mask_threshold: float | None = 0.8,
        ratio_threshold: float = 0.75,
        ransac_reproj_threshold: float = 5.0,
        ransac_max_iters: int = 2000,
        ransac_confidence: float = 0.99,
        rigid: bool = True,
    ) -> None:
        self.method = method
        self.scale = scale
        self.brightness_offset = brightness_offset
        self.brightness_mask_threshold = brightness_mask_threshold
        self.ratio_threshold = ratio_threshold
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.ransac_max_iters = ransac_max_iters
        self.ransac_confidence = ransac_confidence
        self.rigid = rigid
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None

        if method == "sift":
            self._detector = cv2.SIFT_create(
                nfeatures=500, contrastThreshold=0.04, edgeThreshold=10.0, sigma=1.6
            )
            self._norm = cv2.NORM_L2
        elif method == "orb":
            self._detector = cv2.ORB_create(nfeatures=500)
            self._norm = cv2.NORM_HAMMING
        elif method == "akaze":
            self._detector = cv2.AKAZE_create()
            self._norm = cv2.NORM_HAMMING
        else:
            raise ValueError(f"Unsupported method: {method!r} (expected sift, orb or akaze).")

        self._reference_h, self._reference_w = reference.shape[:2]
        reference_gray = self._preprocess(reference)
        # The reference is computed without the brightness mask (matches ari).
        self._keypoints1, descriptors1 = self._detector.detectAndCompute(reference_gray, None)
        if descriptors1 is None or len(descriptors1) == 0:
            raise AlignmentError("Reference image yielded no descriptors.")
        self._descriptors1 = descriptors1

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Grayscale, optionally downscale, CLAHE and brightness-shift."""
        gray = _to_gray(image)
        if self.scale != 1.0:
            gray = cv2.resize(
                gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR
            )
        if self._clahe is not None:
            gray = self._clahe.apply(gray)
        return cv2.convertScaleAbs(gray, alpha=1.0, beta=self.brightness_offset)

    def _remove_scale(self, matrix: np.ndarray) -> np.ndarray:
        """Divide the 2x2 block by its scale so the transform becomes rigid."""
        scale = np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)
        out = matrix.astype(np.float64).copy()
        out[:, :2] = matrix[:, :2] / scale
        return out

    def _rescale_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Correct a reduced-scale affine matrix back to full resolution."""
        if self.scale == 1.0:
            return matrix.astype(np.float64)
        matrix_3x3 = np.vstack([matrix, [0.0, 0.0, 1.0]])
        s = np.diag([self.scale, self.scale, 1.0])
        s_inv = np.linalg.inv(s)
        return (s_inv @ matrix_3x3 @ s)[:2, :]

    def align(self, target: np.ndarray) -> AlignmentResult:
        """Align ``target`` onto the reference frame.

        Args:
            target: Target image, uint8 ``(H, W)`` or ``(H, W, 3)``.

        Returns:
            The aligned image, its full-resolution affine matrix and the
            number of good matches.

        Raises:
            AlignmentError: If the target yields no descriptors, fewer than
                four good matches survive, or RANSAC finds no transform.
        """
        gray = self._preprocess(target)
        keypoints2, descriptors2 = self._detector.detectAndCompute(gray, None)

        if (
            descriptors2 is not None
            and len(keypoints2) > 0
            and self.brightness_mask_threshold is not None
        ):
            threshold_value = int(self.brightness_mask_threshold * 255)
            points = np.round([kp.pt for kp in keypoints2]).astype(int)
            height, width = gray.shape[:2]
            xs = np.clip(points[:, 0], 0, width - 1)
            ys = np.clip(points[:, 1], 0, height - 1)
            keep = gray[ys, xs] <= threshold_value
            indices = np.nonzero(keep)[0]
            keypoints2 = [keypoints2[i] for i in indices]
            descriptors2 = descriptors2[keep]

        if descriptors2 is None or len(descriptors2) == 0:
            raise AlignmentError("Target image yielded no descriptors.")

        matcher = cv2.BFMatcher(self._norm, crossCheck=False)
        knn_matches = matcher.knnMatch(self._descriptors1, descriptors2, k=2)
        good = [
            pair[0]
            for pair in knn_matches
            if len(pair) == 2 and pair[0].distance < self.ratio_threshold * pair[1].distance
        ]
        if len(good) < 4:
            raise AlignmentError(f"Too few good matches: {len(good)} < 4.")

        src_pts = np.array(
            [self._keypoints1[m.queryIdx].pt for m in good], dtype=np.float32
        ).reshape(-1, 1, 2)
        dst_pts = np.array([keypoints2[m.trainIdx].pt for m in good], dtype=np.float32).reshape(
            -1, 1, 2
        )

        # src = reference points, dst = target points, so the matrix maps
        # target coordinates onto the reference frame.
        matrix, _ = cv2.estimateAffinePartial2D(
            dst_pts,
            src_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_threshold,
            maxIters=self.ransac_max_iters,
            confidence=self.ransac_confidence,
        )
        if matrix is None:
            raise AlignmentError("RANSAC failed to estimate an affine transform.")

        if self.rigid:
            matrix = self._remove_scale(matrix)
        matrix_full = self._rescale_matrix(matrix)

        aligned = cv2.warpAffine(target, matrix_full, (self._reference_w, self._reference_h))
        return AlignmentResult(image=aligned, matrix=matrix_full, num_matches=len(good))


def _create_overlay(reference: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    """Build a cyan/magenta overlay of reference vs. aligned image."""
    # Work in int32: adding the two uint8 ink channels would wrap around.
    cyan = 255 - _to_gray(reference).astype(np.int32)
    magenta = 255 - _to_gray(aligned).astype(np.int32)
    red_channel = (255 - magenta).astype(np.uint8)
    green_channel = (255 - cyan).astype(np.uint8)
    blue_channel = (255 - np.minimum(255, cyan + magenta)).astype(np.uint8)
    return cv2.merge((blue_channel, green_channel, red_channel))


def align_directories(
    reference_dir: Path | str,
    target_dir: Path | str,
    output_dir: Path | str,
    *,
    method: str = "sift",
    scale: float = 1.0,
    debug_dir: Path | str | None = None,
) -> list[str]:
    """Align every target image onto its identically-named reference.

    Files are paired strictly by identical file name (the sorted intersection
    of both directories); the ari command paired by sort order, which silently
    mispaired mismatched directories. A pair that cannot be aligned is logged
    and skipped rather than being fatal.

    Args:
        reference_dir: Directory of reference images.
        target_dir: Directory of target images to align.
        output_dir: Destination for the aligned images.
        method: Detector passed to :class:`KeypointAligner`.
        scale: Matching downscale factor.
        debug_dir: If given, writes ``<stem>_affine.json`` (the six affine
            floats) and ``<stem>_overlay.png`` per aligned pair.

    Returns:
        The successfully aligned file names in processing (sorted) order.
    """
    reference_path = Path(reference_dir)
    target_path = Path(target_dir)
    output_path = Path(output_dir)

    reference_names = {p.name for p in reference_path.iterdir() if p.is_file()}
    target_names = {p.name for p in target_path.iterdir() if p.is_file()}
    common = sorted(reference_names & target_names)

    output_path.mkdir(parents=True, exist_ok=True)
    debug_path = Path(debug_dir) if debug_dir is not None else None
    if debug_path is not None:
        debug_path.mkdir(parents=True, exist_ok=True)

    aligned_names: list[str] = []
    for name in common:
        reference = cv2.imread(str(reference_path / name), cv2.IMREAD_COLOR)
        target = cv2.imread(str(target_path / name), cv2.IMREAD_COLOR)
        try:
            aligner = KeypointAligner(reference, method=method, scale=scale)
            result = aligner.align(target)
        except AlignmentError as error:
            logger.warning("Skipping %s: %s", name, error)
            continue

        cv2.imwrite(str(output_path / name), result.image)
        if debug_path is not None:
            stem = Path(name).stem
            payload = {"affine_matrix": result.matrix.flatten().tolist()}
            (debug_path / f"{stem}_affine.json").write_text(json.dumps(payload, indent=4))
            overlay = _create_overlay(reference, result.image)
            cv2.imwrite(str(debug_path / f"{stem}_overlay.png"), overlay)
        aligned_names.append(name)
    return aligned_names
