"""Crop-center sampling and sliding-window tiling (numpy + stdlib only).

Ported from the ari-data-preprocess ``extract`` helpers
(``generate_random_center_points``, ``img_split``) and the ``clips``
command's tiling loop, with a clean typed API.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

import numpy as np

__all__ = ["sample_crop_centers", "crop_centered", "iter_tiles"]


def sample_crop_centers(
    rng: random.Random, width: int, height: int, crop_size: int, count: int
) -> list[tuple[int, int]]:
    """Sample unique integer crop centers that fit fully inside the image.

    A center ``(x, y)`` is valid when the ``crop_size`` square around it lies
    entirely within the image, i.e. ``x in [half, width - crop_size + half]``
    and ``y in [half, height - crop_size + half]`` with ``half = crop_size // 2``.

    Sampling first draws random candidates (up to ``count * 50`` attempts); if
    that does not yield enough unique centers it falls back to an exhaustive
    shuffle of the remaining valid positions. When the number of valid
    positions ("capacity") is smaller than ``count`` all capacity points are
    returned (shuffled). The result is deterministic given ``rng``.

    Args:
        rng: Random generator driving the sampling.
        width: Image width in pixels.
        height: Image height in pixels.
        crop_size: Side length of the square crop.
        count: Desired number of centers.

    Returns:
        A list of unique ``(x, y)`` integer centers.

    Raises:
        ValueError: If the image is smaller than the crop in either dimension.
    """
    if width < crop_size or height < crop_size:
        raise ValueError(f"Image too small: {width}x{height} < {crop_size}x{crop_size}")

    half = crop_size // 2
    x_min, x_max = half, width - crop_size + half
    y_min, y_max = half, height - crop_size + half

    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    max_attempts = count * 50
    attempts = 0
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        point = (rng.randrange(x_min, x_max + 1), rng.randrange(y_min, y_max + 1))
        if point not in seen:
            seen.add(point)
            out.append(point)

    if len(out) < count:
        need = count - len(out)
        valid = [(x, y) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)]
        remaining = [p for p in valid if p not in seen]
        rng.shuffle(remaining)
        out.extend(remaining[:need])
    return out


def crop_centered(
    image: np.ndarray, center: tuple[int, int], crop_size: int, pad_value: int = 0
) -> np.ndarray:
    """Extract a ``crop_size`` square crop centered on ``center = (x, y)``.

    The output always has shape ``(crop_size, crop_size[, C])`` with the same
    dtype and channel count as ``image`` (a 2-D input stays 2-D). Regions that
    fall outside the image are filled with ``pad_value``.

    Args:
        image: Source image, ``(H, W)`` or ``(H, W, C)``.
        center: Crop center ``(x, y)``.
        crop_size: Side length of the square crop.
        pad_value: Fill value for out-of-bounds regions.

    Returns:
        The cropped (and possibly padded) image.
    """
    x, y = center
    src_h, src_w = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    half = crop_size // 2

    x0 = x - half
    x1 = x0 + crop_size
    y0 = y - half
    y1 = y0 + crop_size

    sx0, sx1 = max(0, x0), min(src_w, x1)
    sy0, sy1 = max(0, y0), min(src_h, y1)
    src = image[sy0:sy1, sx0:sx1]

    if channels == 1:
        out = np.full((crop_size, crop_size), pad_value, dtype=image.dtype)
    else:
        out = np.full((crop_size, crop_size, channels), pad_value, dtype=image.dtype)

    dx0 = sx0 - x0
    dy0 = sy0 - y0
    out[dy0 : dy0 + src.shape[0], dx0 : dx0 + src.shape[1]] = src
    return out


def iter_tiles(
    image: np.ndarray, tile_size: tuple[int, int], stride: tuple[int, int]
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield full sliding-window tiles in row-major order.

    Only tiles that fit entirely inside the image are produced; a partial tile
    at the right/bottom edge is dropped, and an image smaller than a single
    tile yields nothing.

    Args:
        image: Source image, ``(H, W)`` or ``(H, W, C)``.
        tile_size: Tile size ``(tile_h, tile_w)``.
        stride: Step ``(stride_y, stride_x)`` between tile origins.

    Yields:
        Tuples ``(y, x, tile)`` with the tile origin and the tile view.
    """
    tile_h, tile_w = tile_size
    stride_y, stride_x = stride
    height, width = image.shape[:2]
    for y in range(0, height - tile_h + 1, stride_y):
        for x in range(0, width - tile_w + 1, stride_x):
            yield y, x, image[y : y + tile_h, x : x + tile_w]
