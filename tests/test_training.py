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

from opencd_lite import CGNet, build_model, load_config  # noqa: E402
from opencd_lite.datasets import BiTemporalFolderDataset  # noqa: E402
from opencd_lite.tasks import (  # noqa: E402
    ChangeDetectionTask,
    HeadLossSpec,
    head_loss_specs_from_config,
)


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


def test_decode_head_is_trained(synthetic_dataset_root: Path, tmp_path: Path, configs_dir) -> None:
    """Models with a parametric decode head train it, not just the backbone.

    SNUNet's config classifies through ``mmseg.FCNHead``; its backbone
    emits a feature map, so a task holding only the backbone can neither
    compute the loss nor update the classifier.
    """
    config = load_config(configs_dir / "snunet" / "snunet_c16_256x256_40k_levircd.py")
    detector = build_model(config)
    detector.train()  # build_model returns an inference-ready (eval) model
    # Derived from the shipped config, exactly as tools/train.py does.
    task = ChangeDetectionTask(
        detector, head_losses=head_loss_specs_from_config(config["model"]), lr=1e-3
    )
    before = detector.decode_head.conv_seg.weight.detach().clone()

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
    trainer.fit(
        task,
        DataLoader(BiTemporalFolderDataset(synthetic_dataset_root), batch_size=2, num_workers=0),
        DataLoader(
            BiTemporalFolderDataset(
                synthetic_dataset_root,
                image_dir_from="val/A",
                image_dir_to="val/B",
                label_dir="val/label",
            ),
            batch_size=1,
            num_workers=0,
        ),
    )
    assert trainer.global_step == 1
    assert not torch.equal(before, detector.decode_head.conv_seg.weight.detach())
    # A 2-class head is scored with argmax, so the metrics stay defined.
    assert "val/f1" in trainer.callback_metrics


def test_head_loss_specs_from_identity_head_config() -> None:
    """CGNet-style config: sigmoid BCE on the final and the auxiliary output."""
    specs = head_loss_specs_from_config(
        {
            "decode_head": {
                "type": "IdentityHead",
                "in_index": -1,
                "out_channels": 1,
                "loss_decode": {"use_sigmoid": True, "loss_weight": 1.0},
            },
            "auxiliary_head": {
                "type": "IdentityHead",
                "in_index": 0,
                "out_channels": 1,
                "loss_decode": {"use_sigmoid": True, "loss_weight": 1.0},
            },
        }
    )
    assert specs == (
        HeadLossSpec(indices=(-1,), use_sigmoid=True, apply_decode_head=False),
        HeadLossSpec(indices=(0,), use_sigmoid=True, apply_decode_head=False),
    )


def test_head_loss_specs_from_deep_supervision_config() -> None:
    """IFN-style config: the auxiliary head consumes a list of outputs."""
    specs = head_loss_specs_from_config(
        {
            "decode_head": {
                "type": "IdentityHead",
                "in_index": -1,
                "loss_decode": {"use_sigmoid": True},
            },
            "auxiliary_head": {
                "type": "DSIdentityHead",
                "in_index": [0, 1, 2, 3],
                "loss_decode": {"use_sigmoid": True},
            },
        }
    )
    assert specs[1].indices == (0, 1, 2, 3)


def test_head_loss_specs_from_parametric_head_config() -> None:
    """FCNHead configs use softmax cross-entropy through the decode head."""
    specs = head_loss_specs_from_config(
        {
            "decode_head": {
                "type": "mmseg.FCNHead",
                "in_index": -1,
                "num_classes": 2,
                "loss_decode": {
                    "use_sigmoid": False,
                    "loss_weight": 2.0,
                    "class_weight": [1.0, 8.0],
                },
            }
        }
    )
    assert specs == (
        HeadLossSpec(
            indices=(-1,),
            use_sigmoid=False,
            loss_weight=2.0,
            class_weight=(1.0, 8.0),
            apply_decode_head=True,
        ),
    )


def test_config_losses_match_the_default_for_identity_heads() -> None:
    """The config-driven loss reduces to the fallback for IdentityHead models."""
    model = CGNet(pretrained=False)
    outputs = model(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
    mask = torch.randint(0, 2, (1, 64, 64))

    specs = head_loss_specs_from_config(
        {
            "decode_head": {
                "type": "IdentityHead",
                "in_index": -1,
                "loss_decode": {"use_sigmoid": True},
            },
            "auxiliary_head": {
                "type": "IdentityHead",
                "in_index": 0,
                "loss_decode": {"use_sigmoid": True},
            },
        }
    )
    with_config = ChangeDetectionTask(model, head_losses=specs)._loss(outputs, mask)
    fallback = ChangeDetectionTask(model)._loss(outputs, mask)
    torch.testing.assert_close(with_config, fallback)


def test_class_weight_changes_the_loss() -> None:
    logits = torch.randn(1, 2, 8, 8)
    mask = torch.zeros(1, 8, 8, dtype=torch.int64)
    mask[..., :2] = 1
    task = ChangeDetectionTask(CGNet(pretrained=False))

    unweighted = task._head_loss(HeadLossSpec(indices=(0,), use_sigmoid=False), [logits], mask)
    weighted = task._head_loss(
        HeadLossSpec(indices=(0,), use_sigmoid=False, class_weight=(1.0, 8.0)), [logits], mask
    )
    assert not torch.isclose(unweighted, weighted)
