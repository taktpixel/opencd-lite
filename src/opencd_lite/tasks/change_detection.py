"""Lightning task for binary change detection.

The task trains a whole :class:`~opencd_lite.inference.ChangeDetector`,
so models whose Open-CD config puts the classifier in a parametric
``decode_head`` (SNUNet, the FC-Siam family) are trained end to end
rather than only up to their backbone.

Losses follow the head sections of the Open-CD config. Each head names
the model outputs it consumes (``in_index``, an int or a list) and the
loss to apply to them (``loss_decode``): sigmoid cross-entropy for
single-channel heads, softmax cross-entropy — optionally class-weighted
— for multi-class ones. Use :func:`head_loss_specs_from_config` to
derive those specs from a loaded config; without them the task falls
back to sigmoid cross-entropy on every model output, which is what the
``IdentityHead`` configurations (CGNet, IFN) declare anyway.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..builder import IDENTITY_HEAD_TYPES
from ..inference import ChangeDetector

__all__ = ["ChangeDetectionTask", "HeadLossSpec", "head_loss_specs_from_config"]

#: mmseg's ignore label (``seg_pad_val`` in the Open-CD configs). Only the
#: softmax path honors it; :class:`~opencd_lite.datasets.BiTemporalFolderDataset`
#: emits binary masks, so nothing is ignored in practice.
IGNORE_INDEX = 255


@dataclass(frozen=True)
class HeadLossSpec:
    """Loss declared by one Open-CD head section.

    Attributes:
        indices: Model outputs this head consumes (``in_index``).
        use_sigmoid: Sigmoid cross-entropy (single-channel head) when
            true, softmax cross-entropy otherwise.
        loss_weight: Weight applied to the head's total loss.
        class_weight: Per-class weights for softmax cross-entropy.
        apply_decode_head: Whether the detector's parametric
            ``decode_head`` is applied to the outputs first. Only the
            ``decode_head`` section does; auxiliary heads read the model
            outputs directly.
    """

    indices: tuple[int, ...] = (-1,)
    use_sigmoid: bool = True
    loss_weight: float = 1.0
    class_weight: tuple[float, ...] | None = None
    apply_decode_head: bool = False


def head_loss_specs_from_config(model_cfg: Mapping[str, Any]) -> tuple[HeadLossSpec, ...]:
    """Derive the loss specs from the ``model`` section of an Open-CD config."""

    def spec(head_cfg: Mapping[str, Any], *, is_decode_head: bool) -> HeadLossSpec:
        loss_cfg: Mapping[str, Any] = head_cfg.get("loss_decode") or {}
        in_index = head_cfg.get("in_index", -1)
        indices = tuple(in_index) if isinstance(in_index, (list, tuple)) else (in_index,)
        class_weight = loss_cfg.get("class_weight")
        return HeadLossSpec(
            indices=indices,
            use_sigmoid=bool(loss_cfg.get("use_sigmoid", False)),
            loss_weight=float(loss_cfg.get("loss_weight", 1.0)),
            class_weight=tuple(class_weight) if class_weight else None,
            apply_decode_head=is_decode_head,
        )

    specs = []
    decode_head = model_cfg.get("decode_head")
    if decode_head:
        # Parameter-free Open-CD heads leave the model output untouched.
        identity = decode_head.get("type", "IdentityHead") in IDENTITY_HEAD_TYPES
        specs.append(spec(decode_head, is_decode_head=not identity))
    auxiliary = model_cfg.get("auxiliary_head")
    if auxiliary:
        for head_cfg in auxiliary if isinstance(auxiliary, list) else [auxiliary]:
            specs.append(spec(head_cfg, is_decode_head=False))
    return tuple(specs)


class ChangeDetectionTask(L.LightningModule):
    """Train a bi-temporal change detection model.

    Args:
        model: A :class:`~opencd_lite.inference.ChangeDetector`, or a
            bare network (e.g. :class:`~opencd_lite.models.CGNet`),
            which is wrapped in a detector with default settings — note
            that those include mmseg's default binarization threshold of
            0.3 for the validation metrics.
        head_losses: Loss specs from the config (see
            :func:`head_loss_specs_from_config`). When omitted, sigmoid
            cross-entropy is applied to every model output; that fallback
            only covers models without a parametric decode head.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.

    Expects batches shaped like
    :class:`~opencd_lite.datasets.BiTemporalFolderDataset` items:
    ``{"image_from": (B,3,H,W), "image_to": (B,3,H,W), "mask": (B,H,W)}``.
    """

    def __init__(
        self,
        model: nn.Module,
        head_losses: Sequence[HeadLossSpec] | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
    ) -> None:
        super().__init__()
        self.model = model if isinstance(model, ChangeDetector) else ChangeDetector(model)
        self.head_losses = tuple(head_losses) if head_losses else ()
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["model", "head_losses"])

    def forward(self, image_from: Tensor, image_to: Tensor) -> Tensor:
        """Prediction logits, as at inference time (backbone + decode head)."""
        return self.model(image_from, image_to)

    # -- loss ---------------------------------------------------------
    def _resize(self, logits: Tensor, target: Tensor) -> Tensor:
        if logits.shape[-2:] == target.shape[-2:]:
            return logits
        return F.interpolate(logits, target.shape[-2:], mode="bilinear", align_corners=False)

    def _head_loss(self, spec: HeadLossSpec, outputs: Sequence[Tensor], mask: Tensor) -> Tensor:
        loss = outputs[0].new_zeros(())
        for index in spec.indices:
            logits = outputs[index]
            if spec.apply_decode_head:
                if self.model.decode_head is None:
                    raise ValueError(
                        "Head loss requests the decode head, but the detector has none. "
                        "Build the model from the same config the specs come from."
                    )
                logits = self.model.decode_head(logits)
            logits = self._resize(logits, mask)
            if spec.use_sigmoid:
                loss = loss + F.binary_cross_entropy_with_logits(logits, mask.float().unsqueeze(1))
            else:
                weight = (
                    logits.new_tensor(spec.class_weight) if spec.class_weight is not None else None
                )
                loss = loss + F.cross_entropy(
                    logits, mask, weight=weight, ignore_index=IGNORE_INDEX
                )
        return spec.loss_weight * loss

    def _loss(self, outputs: Sequence[Tensor], mask: Tensor) -> Tensor:
        if not self.head_losses:
            # No config given: sigmoid cross-entropy on every output, the
            # deep-supervision scheme the IdentityHead configs declare.
            if self.model.decode_head is not None:
                raise ValueError(
                    "This model classifies through a parametric decode head, whose "
                    "loss cannot be inferred. Pass head_losses="
                    "head_loss_specs_from_config(cfg['model'])."
                )
            fallback = HeadLossSpec(indices=tuple(range(len(outputs))), use_sigmoid=True)
            return self._head_loss(fallback, outputs, mask)
        return sum(
            (self._head_loss(spec, outputs, mask) for spec in self.head_losses),
            start=outputs[0].new_zeros(()),
        )

    # -- steps --------------------------------------------------------
    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        outputs = self.model.backbone(batch["image_from"], batch["image_to"])
        loss = self._loss(outputs, batch["mask"])
        self.log("train/loss", loss, prog_bar=True, batch_size=batch["mask"].shape[0])
        return loss

    def on_validation_epoch_start(self) -> None:
        # Confusion-matrix accumulators for the "change" class.
        self._tp = torch.zeros((), dtype=torch.long, device=self.device)
        self._fp = torch.zeros_like(self._tp)
        self._fn = torch.zeros_like(self._tp)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        mask = batch["mask"]
        outputs = self.model.backbone(batch["image_from"], batch["image_to"])
        cfg = self.model.inference_cfg
        logits = outputs[cfg.out_index]
        if self.model.decode_head is not None:
            logits = self.model.decode_head(logits)
        logits = self._resize(logits, mask)
        # Binarize the way the decode head does at test time: a threshold
        # on the sigmoid for single-channel heads, argmax otherwise.
        if cfg.out_channels == 1:
            pred = (logits.sigmoid().squeeze(1) > cfg.threshold).long()
        else:
            pred = logits.argmax(dim=1).long()

        target = mask.long()
        self._tp += ((pred == 1) & (target == 1)).sum()
        self._fp += ((pred == 1) & (target == 0)).sum()
        self._fn += ((pred == 0) & (target == 1)).sum()

        self.log("val/loss", self._loss(outputs, mask), batch_size=mask.shape[0])

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
