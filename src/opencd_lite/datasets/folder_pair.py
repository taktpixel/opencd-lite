"""Folder-based bi-temporal dataset (LEVIR-CD directory layout).

Expected layout (matching the Open-CD ``data_prefix`` convention)::

    <root>/
        <split>/A/       "before" images
        <split>/B/       "after" images
        <split>/label/   binary change masks (0 = unchanged, >0 = changed)

This module is part of the ``train`` extra and requires Pillow.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..transforms import IMAGENET_SPEC, PreprocessSpec, normalize_image

__all__ = ["BiTemporalFolderDataset"]

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

Sample = dict[str, Tensor | str]


class BiTemporalFolderDataset(Dataset[Sample]):
    """Image-pair + mask dataset over parallel directories.

    Args:
        root: Dataset root directory.
        image_dir_from: Sub-path of the "before" images (e.g. ``train/A``).
        image_dir_to: Sub-path of the "after" images (e.g. ``train/B``).
        label_dir: Sub-path of the change masks (e.g. ``train/label``).
        preprocess: Normalization constants applied to both images.
        transform: Optional joint augmentation applied *before*
            normalization; receives and returns
            ``(image_from, image_to, mask)`` uint8 arrays.

    Each item is a dict with keys ``image_from``/``image_to``
    (float32 ``(3, H, W)``), ``mask`` (int64 ``(H, W)`` with values 0/1)
    and ``name`` (file stem).
    """

    def __init__(
        self,
        root: str | Path,
        image_dir_from: str = "train/A",
        image_dir_to: str = "train/B",
        label_dir: str = "train/label",
        preprocess: PreprocessSpec = IMAGENET_SPEC,
        transform: Callable[
            [np.ndarray, np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ]
        | None = None,
    ) -> None:
        self.root = Path(root)
        self.dir_from = self.root / image_dir_from
        self.dir_to = self.root / image_dir_to
        self.dir_label = self.root / label_dir
        self.preprocess = preprocess
        self.transform = transform
        self.names = self._collect_names()

    def _collect_names(self) -> list[str]:
        for directory in (self.dir_from, self.dir_to, self.dir_label):
            if not directory.is_dir():
                raise FileNotFoundError(f"Dataset directory not found: {directory}")
        names = sorted(
            p.name for p in self.dir_from.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not names:
            raise FileNotFoundError(f"No images found under {self.dir_from}")
        missing = [
            name
            for name in names
            if not (self.dir_to / name).is_file() or not (self.dir_label / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} samples lack a pair image or label, e.g. {missing[:3]}"
            )
        return names

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> Sample:
        from PIL import Image  # imported lazily: only needed with the "train" extra

        name = self.names[index]
        image_from = np.asarray(Image.open(self.dir_from / name).convert("RGB"))
        image_to = np.asarray(Image.open(self.dir_to / name).convert("RGB"))
        mask = np.asarray(Image.open(self.dir_label / name).convert("L"))

        if self.transform is not None:
            image_from, image_to, mask = self.transform(image_from, image_to, mask)

        return {
            "image_from": normalize_image(image_from, self.preprocess),
            "image_to": normalize_image(image_to, self.preprocess),
            # LEVIR-CD stores changed pixels as 255; binarize to {0, 1}.
            "mask": torch.from_numpy((mask > 0).astype(np.int64)),
            "name": Path(name).stem,
        }
