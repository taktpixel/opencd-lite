#!/usr/bin/env python
"""Train a change detection model from an Open-CD config.

Reads the model definition, learning rate and data layout from an
upstream-compatible config file and trains with Lightning. Experiment
tracking is recorded to MLflow when a tracking URI is provided (via
``--mlflow-uri`` or the ``MLFLOW_TRACKING_URI`` environment variable);
otherwise metrics fall back to a local CSV logger.

Example:
    python tools/train.py configs/cgnet/cgnet_256x256_40k_levircd.py \
        --data-root /data/LEVIR-CD --max-epochs 100 \
        --mlflow-uri http://localhost:5000 --run-name cgnet-baseline

Requires the ``train`` extra: ``pip install opencd-lite[train]``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import CSVLogger, Logger, MLFlowLogger
from torch.utils.data import DataLoader

from opencd_lite import build_model, load_config
from opencd_lite.datasets import BiTemporalFolderDataset
from opencd_lite.tasks import ChangeDetectionTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Open-CD compatible config file")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root (defaults to the config's data_root)",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("work_dirs"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip ImageNet initialization of the encoder (useful for smoke tests)",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI; enables MLflow logging when set",
    )
    parser.add_argument("--mlflow-experiment", default="opencd-lite")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def make_logger(args: argparse.Namespace) -> Logger:
    if args.mlflow_uri:
        return MLFlowLogger(
            experiment_name=args.mlflow_experiment,
            run_name=args.run_name,
            tracking_uri=args.mlflow_uri,
            log_model=True,
        )
    return CSVLogger(save_dir=str(args.work_dir), name=args.run_name or "default")


def make_dataloaders(
    cfg: dict, data_root: Path, batch_size: int, num_workers: int
) -> tuple[DataLoader, DataLoader]:
    def loader(split_cfg: dict, *, train: bool) -> DataLoader:
        prefix = split_cfg["dataset"]["data_prefix"]
        dataset = BiTemporalFolderDataset(
            root=data_root,
            image_dir_from=prefix["img_path_from"],
            image_dir_to=prefix["img_path_to"],
            label_dir=prefix["seg_map_path"],
        )
        return DataLoader(
            dataset,
            batch_size=batch_size if train else 1,
            shuffle=train,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    return (
        loader(cfg["train_dataloader"], train=True),
        loader(cfg["val_dataloader"], train=False),
    )


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    cfg = load_config(args.config)
    overrides = {"pretrained": False} if args.no_pretrained else None
    detector = build_model(cfg, backbone_overrides=overrides)
    # build_model returns an inference-ready (eval) model; switch to train
    # mode here so Lightning does not warn about frozen submodules.
    detector.train()

    optimizer_cfg = cfg.get("optim_wrapper", {}).get("optimizer", {})
    task = ChangeDetectionTask(
        detector.backbone,
        lr=optimizer_cfg.get("lr", 1e-3),
        weight_decay=optimizer_cfg.get("weight_decay", 0.05),
        threshold=detector.inference_cfg.threshold,
        out_index=detector.inference_cfg.out_index,
    )

    data_root = args.data_root or Path(cfg["train_dataloader"]["dataset"]["data_root"])
    batch_size = args.batch_size or cfg["train_dataloader"].get("batch_size", 8)
    train_loader, val_loader = make_dataloaders(cfg, data_root, batch_size, args.num_workers)

    logger = make_logger(args)
    logger.log_hyperparams(
        {
            "config": str(args.config),
            "model_type": cfg["model"]["backbone"]["type"],
            "batch_size": batch_size,
            "seed": args.seed,
        }
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        default_root_dir=str(args.work_dir),
    )
    trainer.fit(task, train_loader, val_loader)


if __name__ == "__main__":
    main()
