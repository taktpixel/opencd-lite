"""Lightning task for binary change detection with deep supervision.

Reproduces the Open-CD training behavior for the supported models: every
output of the network receives a sigmoid cross-entropy loss against the
change mask (the decode head consumes the final output, the auxiliary
head the remaining deep-supervision outputs, all with weight 1.0).
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["ChangeDetectionTask"]


class ChangeDetectionTask(L.LightningModule):
    """Train a bi-temporal change detection model.

    Args:
        model: Bare network returning a tuple of logit tensors
            (e.g. :class:`~opencd_lite.models.CGNet`).
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        threshold: Sigmoid threshold used for validation metrics.
        out_index: Which output of the model is the primary prediction.

    Expects batches shaped like
    :class:`~opencd_lite.datasets.BiTemporalFolderDataset` items:
    ``{"image_from": (B,3,H,W), "image_to": (B,3,H,W), "mask": (B,H,W)}``.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        threshold: float = 0.5,
        out_index: int = -1,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.threshold = threshold
        self.out_index = out_index
        self.save_hyperparameters(ignore=["model"])

    def forward(self, image_from: Tensor, image_to: Tensor) -> tuple[Tensor, ...]:
        return self.model(image_from, image_to)

    def _deep_supervision_loss(self, outputs: tuple[Tensor, ...], mask: Tensor) -> Tensor:
        """Sum of BCE-with-logits over every output, resized to the mask."""
        target = mask.float().unsqueeze(1)
        loss = torch.zeros((), device=target.device)
        for logits in outputs:
            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.interpolate(
                    logits, target.shape[-2:], mode="bilinear", align_corners=False
                )
            loss = loss + F.binary_cross_entropy_with_logits(logits, target)
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        outputs = self(batch["image_from"], batch["image_to"])
        loss = self._deep_supervision_loss(outputs, batch["mask"])
        self.log("train/loss", loss, prog_bar=True, batch_size=batch["mask"].shape[0])
        return loss

    def on_validation_epoch_start(self) -> None:
        # Confusion-matrix accumulators for the "change" class.
        self._tp = torch.zeros((), dtype=torch.long, device=self.device)
        self._fp = torch.zeros_like(self._tp)
        self._fn = torch.zeros_like(self._tp)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        outputs = self(batch["image_from"], batch["image_to"])
        logits = outputs[self.out_index]
        if logits.shape[-2:] != batch["mask"].shape[-2:]:
            logits = F.interpolate(
                logits, batch["mask"].shape[-2:], mode="bilinear", align_corners=False
            )
        pred = (logits.sigmoid().squeeze(1) > self.threshold).long()
        target = batch["mask"].long()
        self._tp += ((pred == 1) & (target == 1)).sum()
        self._fp += ((pred == 1) & (target == 0)).sum()
        self._fn += ((pred == 0) & (target == 1)).sum()

        loss = self._deep_supervision_loss(outputs, batch["mask"])
        self.log("val/loss", loss, batch_size=batch["mask"].shape[0])

    def on_validation_epoch_end(self) -> None:
        tp, fp, fn = self._tp.float(), self._fp.float(), self._fn.float()
        eps = torch.finfo(torch.float32).eps
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        self.log_dict(
            {
                "val/precision": precision,
                "val/recall": recall,
                "val/f1": f1,
                "val/iou": iou,
            },
            prog_bar=True,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
