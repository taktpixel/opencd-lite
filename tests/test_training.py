"""Training smoke tests (require the "train" extra)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("lightning")
pytestmark = pytest.mark.train

import lightning as L  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from opencd_lite import CGNet  # noqa: E402
from opencd_lite.datasets import BiTemporalFolderDataset  # noqa: E402
from opencd_lite.tasks import ChangeDetectionTask  # noqa: E402


@pytest.fixture()
def synthetic_dataset_root(tmp_path: Path) -> Path:
    """Create a tiny LEVIR-CD-layout dataset with random images."""
    from PIL import Image

    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        for sub in ("A", "B", "label"):
            (tmp_path / split / sub).mkdir(parents=True)
        for i in range(2):
            name = f"sample_{i}.png"
            for sub in ("A", "B"):
                image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
                Image.fromarray(image).save(tmp_path / split / sub / name)
            mask = rng.choice([0, 255], size=(64, 64)).astype(np.uint8)
            Image.fromarray(mask, mode="L").save(tmp_path / split / "label" / name)
    return tmp_path


def test_dataset_items(synthetic_dataset_root: Path) -> None:
    dataset = BiTemporalFolderDataset(synthetic_dataset_root)
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample["image_from"].shape == (3, 64, 64)
    assert sample["image_to"].shape == (3, 64, 64)
    assert sample["mask"].shape == (64, 64)
    assert sample["mask"].dtype == torch.int64
    assert set(sample["mask"].unique().tolist()) <= {0, 1}


def test_one_training_step(synthetic_dataset_root: Path, tmp_path: Path) -> None:
    """Train one step and validate once on synthetic data."""
    train_loader = DataLoader(
        BiTemporalFolderDataset(synthetic_dataset_root), batch_size=2, num_workers=0
    )
    val_loader = DataLoader(
        BiTemporalFolderDataset(
            synthetic_dataset_root,
            image_dir_from="val/A",
            image_dir_to="val/B",
            label_dir="val/label",
        ),
        batch_size=1,
        num_workers=0,
    )

    task = ChangeDetectionTask(CGNet(pretrained=False), lr=1e-4)
    trainer = L.Trainer(
        max_steps=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(task, train_loader, val_loader)
    assert trainer.global_step == 1
    metrics = trainer.callback_metrics
    assert "val/f1" in metrics
