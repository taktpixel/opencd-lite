#!/usr/bin/env python
"""Prepare a LEVIR-CD-layout change-detection dataset from raw image folders.

Wraps the ``opencd_lite.data_prep`` utilities behind a single argparse CLI so
users can go from raw before/after image folders to a dataset consumable by
``BiTemporalFolderDataset`` and ``tools/train.py``.

Subcommands:

* ``align``     keypoint alignment of after-images onto before-images
* ``crop``      random crops + CAR-stratified balanced train/val/test split
* ``split``     shuffle & split parallel directories into train/val
* ``intersect`` keep only file names common to every directory
* ``convert``   re-encode images to another format (default png)
* ``tile``      sliding-window tiling

Only ``align`` needs OpenCV; every other subcommand works with just Pillow.

Examples:
    python tools/prepare_data.py align raw/before raw/after -o aligned
    python tools/prepare_data.py crop --image-from A --image-to B \
        --label label -o dataset --crop-size 256 --count 30
    python tools/prepare_data.py tile scenes -o tiles --tile-size 1024 1024

Requires the ``dataprep`` extra: ``pip install opencd-lite[dataprep]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from opencd_lite.data_prep import (
    build_crop_dataset,
    convert_images,
    intersect_directories,
    split_directories,
    tile_directory,
)


def _cmd_align(args: argparse.Namespace) -> None:
    # Imported lazily so the other subcommands work without OpenCV installed.
    from opencd_lite.data_prep import alignment

    aligned = alignment.align_directories(
        args.reference_dir,
        args.target_dir,
        args.output,
        method=args.method,
        scale=args.scale,
        debug_dir=args.debug_dir,
    )
    print(f"Aligned {len(aligned)} pair(s) into {args.output}")


def _cmd_crop(args: argparse.Namespace) -> None:
    counts = build_crop_dataset(
        args.image_from,
        args.image_to,
        args.label,
        args.output,
        crop_size=args.crop_size,
        crops_per_image=args.count,
        split=tuple(args.split),
        seed=args.seed,
        balance=not args.no_balance,
    )
    summary = ", ".join(f"{split}={n}" for split, n in counts.items())
    print(f"Wrote crop dataset to {args.output} ({summary})")


def _cmd_split(args: argparse.Namespace) -> None:
    counts = split_directories(
        args.dirs,
        args.output,
        ratio=args.ratio,
        seed=args.seed,
        names=tuple(args.names),
    )
    summary = ", ".join(f"{split}={n}" for split, n in counts.items())
    print(f"Split {len(args.dirs)} director(ies) into {args.output} ({summary})")


def _cmd_intersect(args: argparse.Namespace) -> None:
    common = intersect_directories(args.dirs, args.output)
    print(f"Copied {len(common)} common file(s) into {args.output}")


def _cmd_convert(args: argparse.Namespace) -> None:
    written = convert_images(args.input_dir, args.output, pattern=args.pattern, fmt=args.to)
    print(f"Converted {len(written)} image(s) into {args.output}")


def _cmd_tile(args: argparse.Namespace) -> None:
    count = tile_directory(
        args.input_dir,
        args.output,
        pattern=args.pattern,
        tile_size=tuple(args.tile_size),
        stride=tuple(args.stride),
        per_image_subdir=not args.flat,
    )
    print(f"Wrote {count} tile(s) into {args.output}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    align = subparsers.add_parser("align", help="keypoint-align target onto reference images")
    align.add_argument("reference_dir", type=Path, help="directory of reference images")
    align.add_argument("target_dir", type=Path, help="directory of target images to align")
    align.add_argument("-o", "--output", type=Path, required=True, help="output directory")
    align.add_argument("--method", default="sift", help="detector: sift, orb or akaze")
    align.add_argument("--scale", type=float, default=1.0, help="matching downscale factor")
    align.add_argument("--debug-dir", type=Path, default=None, help="write overlay/affine debug")
    align.set_defaults(func=_cmd_align)

    crop = subparsers.add_parser("crop", help="random crops + CAR-stratified split")
    crop.add_argument("--image-from", type=Path, required=True, help='"before" images (A)')
    crop.add_argument("--image-to", type=Path, required=True, help='"after" images (B)')
    crop.add_argument("--label", type=Path, required=True, help="change masks")
    crop.add_argument("-o", "--output", type=Path, required=True, help="output directory")
    crop.add_argument("--crop-size", type=int, default=256, help="square crop side length")
    crop.add_argument("--count", type=int, default=30, help="crop centers per image")
    crop.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=[0.6, 0.2, 0.2],
        metavar=("TRAIN", "VAL", "TEST"),
        help="train/val/test fractions (must sum to 1)",
    )
    crop.add_argument("--seed", type=int, default=0, help="random seed")
    crop.add_argument("--no-balance", action="store_true", help="disable class balancing")
    crop.set_defaults(func=_cmd_crop)

    split = subparsers.add_parser("split", help="shuffle & split parallel directories")
    split.add_argument("dirs", type=Path, nargs="+", help="parallel directories to split")
    split.add_argument("-o", "--output", type=Path, required=True, help="output directory")
    split.add_argument("--ratio", type=float, default=0.8, help="fraction for the first split")
    split.add_argument("--seed", type=int, default=0, help="random seed")
    split.add_argument(
        "--names",
        nargs=2,
        default=["train", "val"],
        metavar=("FIRST", "SECOND"),
        help="split directory names",
    )
    split.set_defaults(func=_cmd_split)

    intersect = subparsers.add_parser("intersect", help="keep only common file names")
    intersect.add_argument("dirs", type=Path, nargs="+", help="directories to intersect")
    intersect.add_argument("-o", "--output", type=Path, required=True, help="output directory")
    intersect.set_defaults(func=_cmd_intersect)

    convert = subparsers.add_parser("convert", help="re-encode images to another format")
    convert.add_argument("input_dir", type=Path, help="directory searched recursively")
    convert.add_argument("-o", "--output", type=Path, required=True, help="output directory")
    convert.add_argument("--pattern", default="*", help="rglob pattern selecting source files")
    convert.add_argument("--to", default="png", help="target format / extension")
    convert.set_defaults(func=_cmd_convert)

    tile = subparsers.add_parser("tile", help="sliding-window tiling")
    tile.add_argument("input_dir", type=Path, help="directory searched recursively")
    tile.add_argument("-o", "--output", type=Path, required=True, help="output directory")
    tile.add_argument("--pattern", default="*.png", help="rglob pattern selecting source files")
    tile.add_argument(
        "--tile-size",
        type=int,
        nargs=2,
        default=[1024, 1024],
        metavar=("H", "W"),
        help="tile size (height width)",
    )
    tile.add_argument(
        "--stride",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("Y", "X"),
        help="stride (y x) between tiles",
    )
    tile.add_argument("--flat", action="store_true", help="write tiles directly, no per-image dir")
    tile.set_defaults(func=_cmd_tile)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
