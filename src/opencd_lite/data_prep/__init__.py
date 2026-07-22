"""Dataset preparation utilities (``dataprep`` extra).

This subpackage ports the ari-data-preprocess tooling into opencd-lite: it
turns raw before/after image folders into a LEVIR-CD-layout dataset that
``BiTemporalFolderDataset`` and ``tools/train.py`` can consume.

Only the cv2-free API is re-exported here so that importing
``opencd_lite.data_prep`` never requires OpenCV. Keypoint-based alignment
lives in :mod:`opencd_lite.data_prep.alignment` and must be imported
explicitly by users who have the ``dataprep`` extra installed.
"""

from __future__ import annotations

from .ops import (
    build_crop_dataset,
    convert_images,
    intersect_directories,
    split_directories,
    tile_directory,
)
from .splitting import change_area_ratio, change_bin, split_list
from .tiling import crop_centered, iter_tiles, sample_crop_centers

__all__ = [
    "sample_crop_centers",
    "crop_centered",
    "iter_tiles",
    "split_list",
    "change_area_ratio",
    "change_bin",
    "convert_images",
    "tile_directory",
    "intersect_directories",
    "split_directories",
    "build_crop_dataset",
]
