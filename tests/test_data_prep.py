"""Tests for the dataset preparation utilities (``opencd_lite.data_prep``).

These tests define the API contract of the data_prep subpackage: tiling,
splitting and directory-level operations that turn raw before/after image
folders into a LEVIR-CD-layout dataset. They run CPU-only and require the
``dataprep`` extra (Pillow). OpenCV-based alignment is covered separately
in ``test_data_prep_alignment.py``.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest

from opencd_lite.data_prep import (
    build_crop_dataset,
    change_area_ratio,
    change_bin,
    convert_images,
    crop_centered,
    intersect_directories,
    iter_tiles,
    sample_crop_centers,
    split_directories,
    split_list,
    tile_directory,
)

pytest.importorskip("PIL")

pytestmark = pytest.mark.dataprep

REPO_ROOT = Path(__file__).resolve().parents[1]


def _save_image(path: Path, array: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _load_image(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img)


def _make_crop_tree(root: Path) -> tuple[Path, Path, Path]:
    """Six 48x48 triplets: 2 all-black masks (bin 0), 4 all-white (bin 3)."""
    rng = np.random.default_rng(0)
    a_dir, b_dir, label_dir = root / "A", root / "B", root / "MASK"
    for i in range(6):
        name = f"img{i}.png"
        _save_image(a_dir / name, rng.integers(0, 255, (48, 48, 3), dtype=np.uint8))
        _save_image(b_dir / name, rng.integers(0, 255, (48, 48, 3), dtype=np.uint8))
        value = 0 if i < 2 else 255
        _save_image(label_dir / name, np.full((48, 48), value, dtype=np.uint8))
    return a_dir, b_dir, label_dir


@pytest.fixture(scope="module")
def cli():
    """The tools/prepare_data.py CLI loaded as a module."""
    spec = importlib.util.spec_from_file_location(
        "prepare_data", REPO_ROOT / "tools" / "prepare_data.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSampleCropCenters:
    def test_centers_unique_and_within_bounds(self):
        rng = random.Random(0)
        centers = sample_crop_centers(rng, width=100, height=80, crop_size=32, count=20)
        assert len(centers) == 20
        assert len(set(centers)) == 20
        half = 16
        for x, y in centers:
            assert half <= x <= 100 - 32 + half
            assert half <= y <= 80 - 32 + half

    def test_deterministic_for_equal_seeds(self):
        first = sample_crop_centers(random.Random(7), 64, 64, 16, 10)
        second = sample_crop_centers(random.Random(7), 64, 64, 16, 10)
        assert first == second

    def test_returns_capacity_when_count_exceeds_positions(self):
        # 33x33 image with 32-pixel crops leaves only 2x2 valid centers.
        centers = sample_crop_centers(random.Random(0), 33, 33, 32, 10)
        assert len(centers) == 4
        assert len(set(centers)) == 4

    def test_too_small_image_raises(self):
        with pytest.raises(ValueError):
            sample_crop_centers(random.Random(0), 16, 64, 32, 1)


class TestCropCentered:
    def test_interior_crop_matches_slice(self):
        values = (np.arange(40 * 50 * 3) % 251).astype(np.uint8)
        image = values.reshape(40, 50, 3)
        crop = crop_centered(image, center=(25, 20), crop_size=10)
        assert crop.shape == (10, 10, 3)
        np.testing.assert_array_equal(crop, image[15:25, 20:30])

    def test_border_crop_is_padded(self):
        image = np.full((32, 32), 9, dtype=np.uint8)
        crop = crop_centered(image, center=(0, 0), crop_size=8, pad_value=0)
        assert crop.shape == (8, 8)
        assert (crop[:4, :] == 0).all()
        assert (crop[:, :4] == 0).all()
        assert (crop[4:, 4:] == 9).all()

    def test_pad_value_and_dtype(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        crop = crop_centered(image, center=(0, 8), crop_size=8, pad_value=7)
        assert crop.dtype == np.uint8
        assert crop.shape == (8, 8, 3)
        assert (crop[:, :4] == 7).all()


class TestIterTiles:
    def test_row_major_full_tiles(self):
        image = np.arange(64 * 96).reshape(64, 96)
        tiles = list(iter_tiles(image, tile_size=(32, 32), stride=(16, 32)))
        assert len(tiles) == 9
        assert [t[0] for t in tiles] == [0, 0, 0, 16, 16, 16, 32, 32, 32]
        assert [t[1] for t in tiles] == [0, 32, 64] * 3
        for y, x, tile in tiles:
            assert tile.shape == (32, 32)
            np.testing.assert_array_equal(tile, image[y : y + 32, x : x + 32])

    def test_partial_tiles_are_dropped(self):
        image = np.zeros((33, 33))
        assert len(list(iter_tiles(image, (32, 32), (32, 32)))) == 1

    def test_image_smaller_than_tile_yields_nothing(self):
        image = np.zeros((16, 16))
        assert list(iter_tiles(image, (32, 32), (32, 32))) == []


class TestSplitList:
    def test_sizes_and_disjoint_union(self):
        items = list(range(10))
        first, second = split_list(random.Random(0), items, 0.8)
        assert len(first) == 8
        assert len(second) == 2
        assert sorted(first + second) == items
        assert items == list(range(10))  # the input list is untouched

    def test_deterministic(self):
        first = split_list(random.Random(3), list(range(20)), 0.5)
        second = split_list(random.Random(3), list(range(20)), 0.5)
        assert first == second

    def test_extreme_ratios(self):
        items = ["a", "b", "c"]
        assert split_list(random.Random(0), items, 0.0)[0] == []
        assert split_list(random.Random(0), items, 1.0)[1] == []


class TestChangeAreaRatio:
    def test_basic_fractions(self):
        assert change_area_ratio(np.zeros((4, 4), dtype=np.uint8)) == 0.0
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[:2] = 255
        assert change_area_ratio(mask) == 0.5

    def test_threshold_is_strict(self):
        mask = np.full((2, 2), 5, dtype=np.uint8)
        assert change_area_ratio(mask, threshold=5) == 0.0
        assert change_area_ratio(mask, threshold=4) == 1.0

    def test_empty_mask(self):
        assert change_area_ratio(np.zeros((0, 0), dtype=np.uint8)) == 0.0


class TestChangeBin:
    @pytest.mark.parametrize(
        ("ratio", "expected"),
        [(0.0, 0), (0.01, 1), (0.05, 1), (0.051, 2), (0.2, 2), (0.5, 3), (1.0, 3)],
    )
    def test_default_thresholds(self, ratio, expected):
        assert change_bin(ratio) == expected

    def test_custom_thresholds(self):
        assert change_bin(0.4, thresholds=(0.1, 0.3, 0.5)) == 3


class TestConvertImages:
    def test_mirrors_relative_tree(self, tmp_path):
        src = tmp_path / "src"
        rng = np.random.default_rng(0)
        img_a = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
        img_b = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
        _save_image(src / "a.bmp", img_a)
        _save_image(src / "sub" / "b.bmp", img_b)
        _save_image(src / "ignored.png", img_a)

        out = tmp_path / "out"
        written = convert_images(src, out, pattern="*.bmp")

        relative = sorted(p.relative_to(out).as_posix() for p in written)
        assert relative == ["a.png", "sub/b.png"]
        np.testing.assert_array_equal(_load_image(out / "a.png"), img_a)
        np.testing.assert_array_equal(_load_image(out / "sub" / "b.png"), img_b)


class TestTileDirectory:
    def test_per_image_subdir(self, tmp_path):
        src = tmp_path / "src"
        image = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        _save_image(src / "big.png", image)

        out = tmp_path / "out"
        count = tile_directory(src, out, pattern="*.png", tile_size=(32, 32), stride=(32, 32))

        assert count == 4
        names = sorted(p.name for p in (out / "big").iterdir())
        assert names == [f"big_{i:04d}.png" for i in range(4)]
        np.testing.assert_array_equal(_load_image(out / "big" / "big_0000.png"), image[:32, :32])

    def test_flat_output(self, tmp_path):
        src = tmp_path / "src"
        _save_image(src / "x.png", np.zeros((32, 64, 3), dtype=np.uint8))
        out = tmp_path / "out"
        count = tile_directory(
            src, out, tile_size=(32, 32), stride=(32, 32), per_image_subdir=False
        )
        assert count == 2
        assert sorted(p.name for p in out.iterdir()) == ["x_0000.png", "x_0001.png"]


class TestIntersectDirectories:
    def test_common_names_copied(self, tmp_path):
        img = np.zeros((4, 4), dtype=np.uint8)
        layout = {
            "A": ["a.png", "b.png", "c.png"],
            "B": ["b.png", "c.png", "d.png"],
            "M": ["b.png", "c.png"],
        }
        for dirname, names in layout.items():
            for name in names:
                _save_image(tmp_path / dirname / name, img)

        out = tmp_path / "out"
        common = intersect_directories([tmp_path / d for d in layout], out)

        assert common == ["b.png", "c.png"]
        for dirname in layout:
            copied = sorted(p.name for p in (out / dirname).iterdir())
            assert copied == ["b.png", "c.png"]

    def test_empty_intersection(self, tmp_path):
        _save_image(tmp_path / "A" / "a.png", np.zeros((4, 4), dtype=np.uint8))
        _save_image(tmp_path / "B" / "b.png", np.zeros((4, 4), dtype=np.uint8))
        result = intersect_directories([tmp_path / "A", tmp_path / "B"], tmp_path / "out")
        assert result == []


class TestSplitDirectories:
    def _make_dirs(self, root: Path, count: int = 10) -> list[Path]:
        dirs = [root / "A", root / "B", root / "MASK"]
        img = np.zeros((4, 4), dtype=np.uint8)
        for i in range(count):
            for directory in dirs:
                _save_image(directory / f"img{i}.png", img)
        return dirs

    def test_ratio_and_pairing(self, tmp_path):
        dirs = self._make_dirs(tmp_path)
        out = tmp_path / "out"
        counts = split_directories(dirs, out, ratio=0.8, seed=0)
        assert counts == {"train": 8, "val": 2}
        for split, n in counts.items():
            reference_names: list[str] | None = None
            for dirname in ("A", "B", "MASK"):
                names = sorted(p.name for p in (out / split / dirname).iterdir())
                assert len(names) == n
                if reference_names is None:
                    reference_names = names
                assert names == reference_names  # pairing is preserved
        train = {p.name for p in (out / "train" / "A").iterdir()}
        val = {p.name for p in (out / "val" / "A").iterdir()}
        assert not train & val

    def test_deterministic(self, tmp_path):
        dirs = self._make_dirs(tmp_path)
        out1, out2 = tmp_path / "o1", tmp_path / "o2"
        split_directories(dirs, out1, ratio=0.5, seed=42)
        split_directories(dirs, out2, ratio=0.5, seed=42)
        first = sorted(p.name for p in (out1 / "train" / "A").iterdir())
        second = sorted(p.name for p in (out2 / "train" / "A").iterdir())
        assert first == second

    def test_mismatched_names_raise(self, tmp_path):
        dirs = self._make_dirs(tmp_path)
        (dirs[1] / "img0.png").unlink()
        with pytest.raises(ValueError):
            split_directories(dirs, tmp_path / "out")


class TestBuildCropDataset:
    def test_unbalanced_counts_and_layout(self, tmp_path):
        a_dir, b_dir, label_dir = _make_crop_tree(tmp_path / "data")
        out = tmp_path / "out"
        counts = build_crop_dataset(
            a_dir,
            b_dir,
            label_dir,
            out,
            crop_size=16,
            crops_per_image=4,
            split=(0.5, 0.25, 0.25),
            seed=0,
            balance=False,
        )
        # 2 all-black masks -> 8 crops in bin 0; 4 all-white -> 16 crops in bin 3.
        # Per bin: round-based split 0.5/0.25/0.25 gives (4,2,2) and (8,4,4).
        assert counts == {"train": 12, "val": 6, "test": 6}
        for split, n in counts.items():
            reference_names: list[str] | None = None
            for sub in ("A", "B", "label"):
                files = sorted(p.name for p in (out / split / sub).iterdir())
                assert len(files) == n
                if reference_names is None:
                    reference_names = files
                assert files == reference_names
        sample = _load_image(next(iter((out / "train" / "A").iterdir())))
        assert sample.shape == (16, 16, 3)
        mask = _load_image(next(iter((out / "train" / "label").iterdir())))
        assert mask.shape == (16, 16)

    def test_balanced_caps_bins(self, tmp_path):
        a_dir, b_dir, label_dir = _make_crop_tree(tmp_path / "data")
        out = tmp_path / "out"
        counts = build_crop_dataset(
            a_dir,
            b_dir,
            label_dir,
            out,
            crop_size=16,
            crops_per_image=4,
            split=(0.5, 0.25, 0.25),
            seed=0,
            balance=True,
        )
        # Both non-empty bins are capped at 8 crops -> 16 total.
        assert counts == {"train": 8, "val": 4, "test": 4}

    def test_zero_fraction_split_is_not_created(self, tmp_path):
        a_dir, b_dir, label_dir = _make_crop_tree(tmp_path / "data")
        out = tmp_path / "out"
        counts = build_crop_dataset(
            a_dir,
            b_dir,
            label_dir,
            out,
            crop_size=16,
            crops_per_image=2,
            split=(0.5, 0.5, 0.0),
            seed=1,
        )
        assert counts["test"] == 0
        assert not (out / "test").exists()
        assert (out / "train" / "label").is_dir()

    def test_invalid_split_raises(self, tmp_path):
        a_dir, b_dir, label_dir = _make_crop_tree(tmp_path / "data")
        with pytest.raises(ValueError):
            build_crop_dataset(a_dir, b_dir, label_dir, tmp_path / "out", split=(0.5, 0.2, 0.2))


class TestPrepareDataCli:
    def test_tile_command(self, tmp_path, cli):
        src = tmp_path / "src"
        _save_image(src / "x.png", np.zeros((64, 64, 3), dtype=np.uint8))
        out = tmp_path / "out"
        cli.main(
            [
                "tile",
                str(src),
                "-o",
                str(out),
                "--tile-size",
                "32",
                "32",
                "--stride",
                "32",
                "32",
                "--flat",
            ]
        )
        assert len(list(out.glob("x_*.png"))) == 4

    def test_convert_command(self, tmp_path, cli):
        src = tmp_path / "src"
        _save_image(src / "a.bmp", np.zeros((8, 8, 3), dtype=np.uint8))
        out = tmp_path / "out"
        cli.main(["convert", str(src), "-o", str(out), "--pattern", "*.bmp"])
        assert (out / "a.png").is_file()

    def test_crop_command(self, tmp_path, cli):
        a_dir, b_dir, label_dir = _make_crop_tree(tmp_path / "data")
        out = tmp_path / "out"
        cli.main(
            [
                "crop",
                "--image-from",
                str(a_dir),
                "--image-to",
                str(b_dir),
                "--label",
                str(label_dir),
                "-o",
                str(out),
                "--crop-size",
                "16",
                "--count",
                "2",
                "--split",
                "0.5",
                "0.5",
                "0.0",
                "--seed",
                "1",
            ]
        )
        assert (out / "train" / "A").is_dir()
        assert (out / "val" / "label").is_dir()
        assert not (out / "test").exists()

    def test_split_command(self, tmp_path, cli):
        img = np.zeros((4, 4), dtype=np.uint8)
        for dirname in ("A", "B"):
            for i in range(4):
                _save_image(tmp_path / dirname / f"{i}.png", img)
        out = tmp_path / "out"
        cli.main(
            [
                "split",
                str(tmp_path / "A"),
                str(tmp_path / "B"),
                "-o",
                str(out),
                "--ratio",
                "0.5",
                "--seed",
                "0",
            ]
        )
        assert len(list((out / "train" / "A").iterdir())) == 2
        assert len(list((out / "val" / "B").iterdir())) == 2

    def test_intersect_command(self, tmp_path, cli):
        img = np.zeros((4, 4), dtype=np.uint8)
        _save_image(tmp_path / "A" / "a.png", img)
        _save_image(tmp_path / "A" / "b.png", img)
        _save_image(tmp_path / "B" / "b.png", img)
        out = tmp_path / "out"
        cli.main(["intersect", str(tmp_path / "A"), str(tmp_path / "B"), "-o", str(out)])
        assert sorted(p.name for p in (out / "A").iterdir()) == ["b.png"]
