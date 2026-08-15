"""Inference wrapper reproducing the Open-CD test-time protocol.

:class:`ChangeDetector` wraps a bare model (CGNet, IFN, ...) with the
preprocessing, whole/sliding-window inference and binarization behavior
that Open-CD applies at test time, while remaining a thin ``nn.Module``
whose ``forward`` is ONNX-export friendly.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .protocol import InferenceConfig
from .transforms import IMAGENET_SPEC, PreprocessSpec, normalize_image, pad_to_divisor

__all__ = ["ChangeDetector", "InferenceConfig"]


class ChangeDetector(nn.Module):
    """Bi-temporal change detector with Open-CD compatible test-time behavior.

    Args:
        model: The bare change detection network. Its ``forward`` must
            accept two ``(B, 3, H, W)`` tensors and return a tuple of
            output tensors — unless ``siamese`` is set, in which case it
            is applied to each image separately and must accept a single
            ``(B, 3, H, W)`` tensor.
        preprocess: Normalization/padding specification.
        inference: Test-time protocol (mode, output selection,
            binarization).
        decode_head: Optional parametric head (e.g.
            :class:`~opencd_lite.models.ConvSegHead`) applied to the
            selected model output. ``None`` when the model emits logits
            directly (Open-CD ``IdentityHead`` configurations).
        neck: Parameter-free fusion of the two feature pyramids (e.g.
            :class:`~opencd_lite.models.FeatureFusionNeck`). Required
            when ``siamese`` is set.
        siamese: Run the (shared-weight) model on each image separately
            and fuse the results with ``neck`` — Open-CD's
            ``SiamEncoderDecoder`` layout. When False the model receives
            both images at once (``DIEncoderDecoder`` layout).
    """

    def __init__(
        self,
        model: nn.Module,
        preprocess: PreprocessSpec = IMAGENET_SPEC,
        inference: InferenceConfig | None = None,
        decode_head: nn.Module | None = None,
        neck: nn.Module | None = None,
        siamese: bool = False,
    ) -> None:
        super().__init__()
        if siamese and neck is None:
            raise ValueError("siamese=True requires a neck to fuse the feature pyramids")
        # Attribute names mirror the Open-CD checkpoint key layout
        # ("backbone.*", "decode_head.*") so checkpoints load directly.
        self.backbone = model
        self.decode_head = decode_head
        self.neck = neck
        self.siamese = siamese
        self.preprocess = preprocess
        self.inference_cfg = inference if inference is not None else InferenceConfig()

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Single forward pass returning the prediction logits.

        Inputs must already be normalized and shaped ``(B, 3, H, W)`` with
        ``H``/``W`` compatible with the model. This is the graph exported
        to ONNX.
        """
        if self.siamese:
            assert self.neck is not None
            outputs = self.neck(self.backbone(x1), self.backbone(x2))
        else:
            outputs = self.backbone(x1, x2)
            if self.neck is not None:
                # Dual-input layout with a pyramid-refining neck
                # (e.g. TinyFPN on LightCDNet features).
                outputs = self.neck(outputs)
        out_index = self.inference_cfg.out_index
        if out_index is None:
            # The decode head consumes the full output tuple itself
            # (e.g. DS_FPNHead drops the early feature internally).
            assert self.decode_head is not None, "out_index=None requires a decode head"
            logits = self.decode_head(list(outputs))
        elif isinstance(out_index, tuple):
            # Multi-input decode head (e.g. Changer): feed the selected
            # feature maps as a list.
            assert self.decode_head is not None, "multi-input out_index requires a decode head"
            logits = self.decode_head([outputs[i] for i in out_index])
        else:
            logits = outputs[out_index]
            if self.decode_head is not None:
                logits = self.decode_head(logits)
        if logits.shape[-2:] != x1.shape[-2:]:
            # mmseg resizes head outputs to the input resolution at test
            # time; heads already emitting full resolution are untouched.
            logits = F.interpolate(
                logits,
                size=x1.shape[-2:],
                mode="bilinear",
                align_corners=getattr(self.decode_head, "align_corners", False),
            )
        return logits

    @torch.inference_mode()
    def predict_logits(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Full test-time inference (padding + whole/slide) on normalized batches.

        Returns logits of shape ``(B, out_channels, H, W)`` matching the
        input spatial size.
        """
        if self.inference_cfg.mode == "slide":
            return self._slide_inference(x1, x2)
        return self._whole_inference(x1, x2)

    def _whole_inference(self, x1: Tensor, x2: Tensor) -> Tensor:
        divisor = self.preprocess.size_divisor
        x1, (height, width) = pad_to_divisor(x1, divisor)
        x2, _ = pad_to_divisor(x2, divisor)
        logits = self.forward(x1, x2)
        return logits[..., :height, :width]

    def _slide_inference(self, x1: Tensor, x2: Tensor) -> Tensor:
        cfg = self.inference_cfg
        assert cfg.crop_size is not None and cfg.stride is not None
        h_crop, w_crop = cfg.crop_size
        h_stride, w_stride = cfg.stride

        # Upstream pads the *whole* image to the size divisor before
        # sliding (mmseg does it in the data preprocessor at test time),
        # so the window grid must be laid out on the padded extent —
        # otherwise the bottom/right windows sit at different origins and
        # the predictions differ along those edges.
        divisor = self.preprocess.size_divisor
        x1, (height, width) = pad_to_divisor(x1, divisor)
        x2, _ = pad_to_divisor(x2, divisor)
        batch, _, h_img, w_img = x1.shape

        # Images smaller than the window fall back to whole inference.
        if h_img <= h_crop and w_img <= w_crop:
            return self.forward(x1, x2)[..., :height, :width]

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = x1.new_zeros((batch, cfg.out_channels, h_img, w_img))
        count = x1.new_zeros((batch, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1_pos = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2_pos = min(x1_pos + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1_pos = max(x2_pos - w_crop, 0)
                crop_logits = self._whole_inference(
                    x1[:, :, y1:y2, x1_pos:x2_pos],
                    x2[:, :, y1:y2, x1_pos:x2_pos],
                )
                preds[:, :, y1:y2, x1_pos:x2_pos] += crop_logits
                count[:, :, y1:y2, x1_pos:x2_pos] += 1
        assert torch.all(count > 0), "Sliding window failed to cover the image"
        return (preds / count)[..., :height, :width]

    @torch.inference_mode()
    def predict(
        self,
        image_from: np.ndarray | Tensor,
        image_to: np.ndarray | Tensor,
    ) -> np.ndarray:
        """Predict a binary change mask for a single image pair.

        Args:
            image_from: "Before" image, ``(H, W, 3)`` RGB uint8 array (or
                float array/tensor on the 0-255 scale).
            image_to: "After" image with the same shape.

        Returns:
            ``(H, W)`` uint8 array with 1 marking changed pixels.
        """
        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device
            x1 = normalize_image(image_from, self.preprocess)[None].to(device)
            x2 = normalize_image(image_to, self.preprocess)[None].to(device)
            if x1.shape != x2.shape:
                raise ValueError(
                    f"Image pair shape mismatch: {tuple(x1.shape)} vs {tuple(x2.shape)}"
                )
            logits = self.predict_logits(x1, x2)
            cfg = self.inference_cfg
            if cfg.out_channels == 1:
                mask = (logits.sigmoid() > cfg.threshold).squeeze(0).squeeze(0)
            else:
                mask = logits.argmax(dim=1).squeeze(0)
            return mask.to(torch.uint8).cpu().numpy()
        finally:
            self.train(was_training)
