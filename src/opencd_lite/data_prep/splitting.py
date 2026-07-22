"""List splitting and change-area-ratio helpers (numpy + stdlib only).

Ported from the ari-data-preprocess ``extract`` helpers
(``shuffle_train_val``, ``calc_car``) plus the CAR-based binning that the
old ``extract`` command performed inline.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar

import numpy as np

__all__ = ["split_list", "change_area_ratio", "change_bin"]

T = TypeVar("T")


def split_list(rng: random.Random, items: Sequence[T], ratio: float) -> tuple[list[T], list[T]]:
    """Shuffle a copy of ``items`` and split it into two parts.

    The input sequence is left untouched. The first part receives
    ``int(len(items) * ratio)`` elements, the second gets the rest.

    Args:
        rng: Random generator driving the shuffle.
        items: Items to split (not mutated).
        ratio: Fraction in ``[0, 1]`` going to the first part.

    Returns:
        A ``(first, second)`` tuple of disjoint lists whose union is ``items``.
    """
    shuffled = list(items)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * ratio)
    return shuffled[:cut], shuffled[cut:]


def change_area_ratio(mask: np.ndarray, threshold: int = 0) -> float:
    """Return the fraction of pixels strictly greater than ``threshold``.

    This is the "change area ratio" (CAR): changed pixels over total pixels.
    An empty array returns ``0.0``.

    Args:
        mask: Change mask array.
        threshold: Pixels ``> threshold`` count as changed.

    Returns:
        The changed-pixel fraction in ``[0, 1]``.
    """
    total = mask.size
    if total == 0:
        return 0.0
    changed = int((mask > threshold).sum())
    return changed / total


def change_bin(ratio: float, thresholds: Sequence[float] = (0.05, 0.2)) -> int:
    """Bucket a change-area ratio into an ordinal bin.

    Bin ``0`` means "no change" (``ratio <= 0``); otherwise the bin is
    ``1 + sum(ratio > t for t in thresholds)``. With the default thresholds
    ``(0.05, 0.2)`` the mapping is ``0 -> 0``, ``(0, 0.05] -> 1``,
    ``(0.05, 0.2] -> 2``, ``(0.2, 1] -> 3``.

    Args:
        ratio: Change-area ratio.
        thresholds: Ascending cut points separating the non-empty bins.

    Returns:
        The ordinal bin index.
    """
    if ratio <= 0:
        return 0
    return 1 + sum(ratio > t for t in thresholds)
