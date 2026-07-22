"""Torch-free ONNX inference.

:class:`ONNXChangeDetector` reproduces the full Open-CD test-time
protocol — ImageNet normalization, size-divisor padding, whole or
sliding-window inference, and binarization — on top of an exported ONNX
graph, using only ``numpy`` and ``onnxruntime``. No PyTorch is needed at
deployment time, so ``pip install opencd-lite[onnx]`` is enough to run
inference.

The exported graph is the model's single ``forward`` pass (see
:func:`opencd_lite.export.export_onnx`); the surrounding protocol lives
here and mirrors :class:`opencd_lite.inference.ChangeDetector` step for
step, so both backends produce identical masks for the same weights.

The inference protocol (mode, crop size, stride, threshold, ...) is read
from the ONNX file's metadata when :meth:`ONNXChangeDetector.from_file`
is used — :func:`export_onnx` embeds it there — or can be passed
explicitly. Normalization constants are never embedded; they always come
from :data:`opencd_lite.transforms.IMAGENET_SPEC`, the single source of
truth for preprocessing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .protocol import InferenceConfig
from .transforms import IMAGENET_SPEC, PreprocessSpec, normalize_image_numpy, pad_to_divisor_numpy

if TYPE_CHECKING:
    import onnxruntime as ort

__all__ = ["ONNXChangeDetector", "ONNX_METADATA_KEY"]

#: ONNX metadata key under which the inference protocol is stored.
ONNX_METADATA_KEY = "opencd_lite_inference"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class ONNXChangeDetector:
    """Run a bi-temporal change detector exported to ONNX, without torch.

    Args:
        onnx_path: Path to the exported ``.onnx`` graph.
        inference: Test-time protocol. When ``None``, it is read from the
            ONNX metadata (written by :func:`export_onnx`); an error is
            raised if neither is available.
        preprocess: Normalization/padding specification. Defaults to the
            ImageNet specification every supported model uses.
        providers: onnxruntime execution providers. Defaults to CPU.

    Example::

        from opencd_lite.onnx import ONNXChangeDetector
        import numpy as np
        from PIL import Image

        detector = ONNXChangeDetector.from_file("cgnet.onnx")
        before = np.asarray(Image.open("before.png").convert("RGB"))
        after = np.asarray(Image.open("after.png").convert("RGB"))
        mask = detector.predict(before, after)  # (H, W) uint8, 1 = changed
    """

    def __init__(
        self,
        onnx_path: str | Path,
        inference: InferenceConfig | None = None,
        preprocess: PreprocessSpec = IMAGENET_SPEC,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort  # imported lazily: only needed with the "onnx" extra

        self.onnx_path = Path(onnx_path)
        self.session: ort.InferenceSession = ort.InferenceSession(
            str(self.onnx_path),
            providers=providers or ["CPUExecutionProvider"],
        )
        self.preprocess = preprocess
        embedded = _read_metadata(self.session)
        if inference is None:
            if embedded is None:
                raise ValueError(
                    f"{self.onnx_path} has no embedded inference metadata; pass "
                    "`inference=InferenceConfig(...)` explicitly."
                )
            inference = embedded
        self.inference_cfg = inference
        inputs = self.session.get_inputs()
        self._input_names = [inp.name for inp in inputs]
        if len(self._input_names) != 2:
            raise ValueError(
                f"Expected a 2-input graph (image_from, image_to), got {self._input_names}"
            )
        # The exported graph is fixed-size; remember it to give a clear
        # error instead of an opaque onnxruntime shape failure when a
        # window/whole input does not match (see _check_window_size).
        shape = inputs[0].shape
        self._graph_hw: tuple[int, int] | None = (
            (int(shape[2]), int(shape[3]))
            if len(shape) == 4 and all(isinstance(shape[i], int) for i in (2, 3))
            else None
        )

    @classmethod
    def from_file(
        cls,
        onnx_path: str | Path,
        preprocess: PreprocessSpec = IMAGENET_SPEC,
        providers: list[str] | None = None,
    ) -> ONNXChangeDetector:
        """Build from an ONNX file that carries its inference protocol in metadata."""
        return cls(onnx_path, inference=None, preprocess=preprocess, providers=providers)

    # -- raw graph ----------------------------------------------------
    def _check_window_size(self, height: int, width: int) -> None:
        """Fail clearly when a graph input does not match the fixed export size."""
        if self._graph_hw is not None and (height, width) != self._graph_hw:
            gh, gw = self._graph_hw
            raise ValueError(
                f"This ONNX graph is fixed-size ({gh}x{gw}) but received a "
                f"{height}x{width} input. The graph was exported at a fixed size: "
                "use slide mode with crop_size equal to the export size (it tiles "
                "larger images), feed inputs of exactly the export size in whole "
                "mode, or re-export with the size you need."
            )

    def _run(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        self._check_window_size(x1.shape[-2], x1.shape[-1])
        feeds = {
            self._input_names[0]: x1.astype(np.float32),
            self._input_names[1]: x2.astype(np.float32),
        }
        (logits,) = self.session.run(None, feeds)
        return np.asarray(logits)

    # -- inference protocol (mirrors ChangeDetector) ------------------
    def predict_logits(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Full test-time inference on normalized ``(N, 3, H, W)`` batches."""
        if self.inference_cfg.mode == "slide":
            return self._slide_inference(x1, x2)
        return self._whole_inference(x1, x2)

    def _whole_inference(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        divisor = self.preprocess.size_divisor
        x1, (height, width) = pad_to_divisor_numpy(x1, divisor)
        x2, _ = pad_to_divisor_numpy(x2, divisor)
        logits = self._run(x1, x2)
        return logits[..., :height, :width]

    def _slide_inference(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        cfg = self.inference_cfg
        assert cfg.crop_size is not None and cfg.stride is not None
        h_crop, w_crop = cfg.crop_size
        h_stride, w_stride = cfg.stride

        # Pad the whole image first, then slide — matching mmseg (and the
        # PyTorch ChangeDetector): laying the grid out on the unpadded
        # image would move the bottom/right windows.
        divisor = self.preprocess.size_divisor
        x1, (height, width) = pad_to_divisor_numpy(x1, divisor)
        x2, _ = pad_to_divisor_numpy(x2, divisor)
        batch, _, h_img, w_img = x1.shape

        if h_img <= h_crop and w_img <= w_crop:
            return self._run(x1, x2)[..., :height, :width]

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = np.zeros((batch, cfg.out_channels, h_img, w_img), dtype=np.float32)
        count = np.zeros((batch, 1, h_img, w_img), dtype=np.float32)
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1_pos = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2_pos = min(x1_pos + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1_pos = max(x2_pos - w_crop, 0)
                crop_logits = self._run(
                    x1[:, :, y1:y2, x1_pos:x2_pos], x2[:, :, y1:y2, x1_pos:x2_pos]
                )
                preds[:, :, y1:y2, x1_pos:x2_pos] += crop_logits
                count[:, :, y1:y2, x1_pos:x2_pos] += 1
        assert np.all(count > 0), "Sliding window failed to cover the image"
        return (preds / count)[..., :height, :width]

    def predict(self, image_from: np.ndarray, image_to: np.ndarray) -> np.ndarray:
        """Predict a binary change mask for a single image pair.

        Args:
            image_from: "Before" image, ``(H, W, 3)`` RGB uint8 array (or
                float array on the 0-255 scale).
            image_to: "After" image with the same shape.

        Returns:
            ``(H, W)`` uint8 array with 1 marking changed pixels.
        """
        x1 = normalize_image_numpy(image_from, self.preprocess)[None]
        x2 = normalize_image_numpy(image_to, self.preprocess)[None]
        if x1.shape != x2.shape:
            raise ValueError(f"Image pair shape mismatch: {x1.shape} vs {x2.shape}")
        logits = self.predict_logits(x1, x2)
        cfg = self.inference_cfg
        if cfg.out_channels == 1:
            mask = _sigmoid(logits) > cfg.threshold
            mask = mask[0, 0]
        else:
            mask = logits[0].argmax(axis=0)
        return mask.astype(np.uint8)


def _read_metadata(session: ort.InferenceSession) -> InferenceConfig | None:
    """Read the embedded inference protocol from an ONNX session, if present."""
    metadata = session.get_modelmeta()
    raw: dict[str, str] = getattr(metadata, "custom_metadata_map", {}) or {}
    payload = raw.get(ONNX_METADATA_KEY)
    if not payload:
        return None
    try:
        data: dict[str, Any] = json.loads(payload)
    except json.JSONDecodeError as error:
        import warnings

        warnings.warn(
            f"Ignoring unparseable {ONNX_METADATA_KEY} ONNX metadata: {error}",
            stacklevel=2,
        )
        return None
    return InferenceConfig.from_dict(data)
