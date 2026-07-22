"""Directory-level dataset preparation operations (PIL imported lazily).

These functions turn raw before/after/label folders into a LEVIR-CD-layout
dataset. They deliberately never import cv2; image I/O uses Pillow, imported
lazily inside each function (matching the established lazy-PIL pattern in
``datasets/folder_pair.py``), so this module is usable with only the
``pillow`` half of the ``dataprep`` extra installed.

Ported from the ari-data-preprocess ``cmd`` handlers (``to_png_cmd``,
``clip_img_cmd``, ``minset_cmd``, ``trainval_cmd``, ``all_extract_file_cmd``)
with several correctness fixes noted per function.
"""

from __future__ import annotations

import logging
import random
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .splitting import change_area_ratio, change_bin, split_list
from .tiling import crop_centered, iter_tiles, sample_crop_centers

__all__ = [
    "convert_images",
    "tile_directory",
    "intersect_directories",
    "split_directories",
    "build_crop_dataset",
]

logger = logging.getLogger(__name__)


def convert_images(
    input_dir: Path | str, output_dir: Path | str, pattern: str = "*", fmt: str = "png"
) -> list[Path]:
    """Re-encode every matching image under ``input_dir`` to ``fmt``.

    Each file matched by ``input_dir.rglob(pattern)`` is re-saved under
    ``output_dir`` at the same relative path but with a ``.{fmt}`` suffix.
    Mirroring the relative subtree is a deliberate improvement over the ari
    ``topng`` command, whose flat output silently overwrote same-stem files
    from different subdirectories. Unreadable files are skipped with a warning.

    Args:
        input_dir: Directory searched recursively.
        output_dir: Destination root.
        pattern: ``rglob`` pattern selecting source files.
        fmt: Target image format / extension.

    Returns:
        Sorted list of written paths.
    """
    from PIL import Image, UnidentifiedImageError

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    written: list[Path] = []
    for src in input_path.rglob(pattern):
        if not src.is_file():
            continue
        relative = src.relative_to(input_path).with_suffix(f".{fmt}")
        destination = output_path / relative
        try:
            with Image.open(src) as image:
                image.load()
        except (UnidentifiedImageError, OSError):
            logger.warning("Skipping unreadable image: %s", src)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)
        written.append(destination)
    return sorted(written)


def tile_directory(
    input_dir: Path | str,
    output_dir: Path | str,
    pattern: str = "*.png",
    tile_size: tuple[int, int] = (1024, 1024),
    stride: tuple[int, int] = (512, 512),
    per_image_subdir: bool = True,
) -> int:
    """Cut every matching image into sliding-window tiles.

    Tiles are named ``{stem}_{index:04d}{source_suffix}`` where ``index``
    counts row-major from ``0`` per source image. With ``per_image_subdir``
    the tiles of each source go to ``output_dir/<stem>/``; otherwise they are
    written directly under ``output_dir``.

    Args:
        input_dir: Directory searched recursively.
        output_dir: Destination root.
        pattern: ``rglob`` pattern selecting source files.
        tile_size: Tile size ``(tile_h, tile_w)``.
        stride: Step ``(stride_y, stride_x)`` between tiles.
        per_image_subdir: Whether to nest each source's tiles in its own dir.

    Returns:
        Total number of tiles written.
    """
    from PIL import Image, UnidentifiedImageError

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    total = 0
    for src in sorted(input_path.rglob(pattern)):
        if not src.is_file():
            continue
        try:
            with Image.open(src) as image:
                array = np.asarray(image)
        except (UnidentifiedImageError, OSError):
            logger.warning("Skipping unreadable image: %s", src)
            continue
        target_dir = output_path / src.stem if per_image_subdir else output_path
        target_dir.mkdir(parents=True, exist_ok=True)
        for index, (_, _, tile) in enumerate(iter_tiles(array, tile_size, stride)):
            destination = target_dir / f"{src.stem}_{index:04d}{src.suffix}"
            Image.fromarray(tile).save(destination)
            total += 1
    return total


def intersect_directories(dirs: Sequence[Path | str], output_dir: Path | str) -> list[str]:
    """Copy only the files whose names are common to every directory.

    The intersection is taken over file *names* across all directories; the
    ari ``minset`` command instead kept every file present in the smallest
    directory, which broke pairing when that directory contained an extra name.
    Each common file is copied (``shutil.copy2``) to
    ``output_dir/<dir.name>/<file>``.

    Args:
        dirs: Directories to intersect.
        output_dir: Destination root.

    Returns:
        Sorted list of common file names. An empty intersection creates no
        output directories and returns ``[]``.
    """
    output_path = Path(output_dir)
    directories = [Path(d) for d in dirs]
    name_sets = [{p.name for p in d.iterdir() if p.is_file()} for d in directories]
    if not name_sets:
        return []
    common = set.intersection(*name_sets)
    if not common:
        return []
    common_sorted = sorted(common)
    for directory in directories:
        destination_dir = output_path / directory.name
        destination_dir.mkdir(parents=True, exist_ok=True)
        for name in common_sorted:
            shutil.copy2(directory / name, destination_dir / name)
    return common_sorted


def split_directories(
    dirs: Sequence[Path | str],
    output_dir: Path | str,
    ratio: float = 0.8,
    seed: int = 0,
    names: tuple[str, str] = ("train", "val"),
) -> dict[str, int]:
    """Split parallel directories into two splits, preserving pairing.

    Every directory must contain the same set of file names. The shared name
    set is shuffled and split with ``split_list``; the corresponding files in
    each directory are copied to ``output_dir/<split>/<dir.name>/<file>`` so
    that pairing across directories is preserved.

    Args:
        dirs: Parallel directories with identical file-name sets.
        output_dir: Destination root.
        ratio: Fraction of names going to the first split.
        seed: Seed for the shuffle.
        names: ``(first_split, second_split)`` directory names.

    Returns:
        Mapping ``{split_name: group_count}``.

    Raises:
        ValueError: If the directories do not share an identical name set.
    """
    output_path = Path(output_dir)
    directories = [Path(d) for d in dirs]
    name_sets = [{p.name for p in d.iterdir() if p.is_file()} for d in directories]
    if any(s != name_sets[0] for s in name_sets):
        raise ValueError("All directories must contain the same set of file names.")

    sorted_names = sorted(name_sets[0])
    first, second = split_list(random.Random(seed), sorted_names, ratio)
    groups = {names[0]: first, names[1]: second}
    for split, group in groups.items():
        for directory in directories:
            destination_dir = output_path / split / directory.name
            destination_dir.mkdir(parents=True, exist_ok=True)
            for name in group:
                shutil.copy2(directory / name, destination_dir / name)
    return {split: len(group) for split, group in groups.items()}


def build_crop_dataset(
    image_dir_from: Path | str,
    image_dir_to: Path | str,
    label_dir: Path | str,
    output_dir: Path | str,
    *,
    crop_size: int = 256,
    crops_per_image: int = 30,
    split: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
    balance: bool = True,
    thresholds: Sequence[float] = (0.05, 0.2),
) -> dict[str, int]:
    """Build a cropped, CAR-stratified train/val/test change-detection dataset.

    Port of the ari ``extract`` command, redesigned for correctness. Crops are
    sampled per source triplet, stratified by their label's change-area ratio
    (via :func:`change_bin`), optionally class-balanced, then split into fixed
    ``("train", "val", "test")`` splits and written in the LEVIR-CD layout
    ``output_dir/<split>/{A,B,label}/{stem}_{i:02d}.png``.

    Args:
        image_dir_from: "Before" images (A).
        image_dir_to: "After" images (B).
        label_dir: Change masks.
        output_dir: Destination root.
        crop_size: Side length of the square crops.
        crops_per_image: Number of crop centers sampled per source triplet.
        split: ``(train, val, test)`` fractions; must sum to ``1``.
        seed: Seed for sampling, balancing and splitting.
        balance: Cap every non-empty bin at the smallest non-empty bin count.
            Unlike ari (which took the min over all bins, including empty ones,
            and could zero out the dataset) only non-empty bins are considered.
        thresholds: CAR cut points passed to :func:`change_bin`.

    Returns:
        Mapping ``{"train": n, "val": n, "test": n}`` of crop-triplet counts.

    Raises:
        ValueError: If the three directories differ in file names, or if
            ``split`` does not sum to ``1`` (tolerance ``1e-6``).
    """
    from PIL import Image

    dir_from = Path(image_dir_from)
    dir_to = Path(image_dir_to)
    dir_label = Path(label_dir)
    output_path = Path(output_dir)

    name_sets = [
        {p.name for p in d.iterdir() if p.is_file()} for d in (dir_from, dir_to, dir_label)
    ]
    if name_sets[0] != name_sets[1] or name_sets[0] != name_sets[2]:
        raise ValueError("A/B/label directories must contain the same set of file names.")
    if abs(sum(split) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1, got {split} (sum {sum(split)}).")

    split_names = ("train", "val", "test")
    rng = random.Random(seed)
    names = sorted(name_sets[0])

    # Pass 1: sample crop centers per triplet and bin them by the label CAR.
    # change_bin returns 0..(1 + len(thresholds)), hence 2 + len(thresholds) bins.
    num_bins = 2 + len(thresholds)
    bins: list[list[tuple[str, int, tuple[int, int]]]] = [[] for _ in range(num_bins)]
    for name in names:
        with Image.open(dir_from / name) as img:
            width_a, height_a = img.size
        with Image.open(dir_to / name) as img:
            width_b, height_b = img.size
        with Image.open(dir_label / name) as img:
            label = np.asarray(img.convert("L"))
        height_l, width_l = label.shape[:2]
        width = min(width_a, width_b, width_l)
        height = min(height_a, height_b, height_l)
        centers = sample_crop_centers(rng, width, height, crop_size, crops_per_image)
        for index, center in enumerate(centers):
            crop = crop_centered(label, center, crop_size)
            bin_index = change_bin(change_area_ratio(crop), thresholds)
            bins[bin_index].append((name, index, center))

    # Optional class balancing: cap every non-empty bin at the smallest
    # non-empty bin's size.
    if balance:
        non_empty = [len(b) for b in bins if b]
        if non_empty:
            cap = min(non_empty)
            for records in bins:
                if records:
                    rng.shuffle(records)
                    del records[cap:]

    # Split each bin independently by cumulative rounding so a 0.0 fraction
    # yields exactly zero items.
    assigned: dict[str, list[tuple[str, int, tuple[int, int]]]] = {s: [] for s in split_names}
    for records in bins:
        if not records:
            continue
        rng.shuffle(records)
        n = len(records)
        b1 = round(n * split[0])
        b2 = round(n * (split[0] + split[1]))
        assigned["train"].extend(records[:b1])
        assigned["val"].extend(records[b1:b2])
        assigned["test"].extend(records[b2:])

    # Pass 2: load each triplet once and write out the selected crops.
    by_name: dict[str, list[tuple[str, int, tuple[int, int]]]] = {}
    for split_name, records in assigned.items():
        for name, index, center in records:
            by_name.setdefault(name, []).append((split_name, index, center))

    for name, entries in by_name.items():
        stem = Path(name).stem
        image_a = np.asarray(Image.open(dir_from / name).convert("RGB"))
        image_b = np.asarray(Image.open(dir_to / name).convert("RGB"))
        label = np.asarray(Image.open(dir_label / name).convert("L"))
        for split_name, index, center in entries:
            for sub, source in (("A", image_a), ("B", image_b), ("label", label)):
                destination_dir = output_path / split_name / sub
                destination_dir.mkdir(parents=True, exist_ok=True)
                crop = crop_centered(source, center, crop_size)
                Image.fromarray(crop).save(destination_dir / f"{stem}_{index:02d}.png")

    return {split_name: len(records) for split_name, records in assigned.items()}
