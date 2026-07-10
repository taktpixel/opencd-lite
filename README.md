# opencd-lite

Lightweight, mmlab-free PyTorch implementation of [Open-CD](https://github.com/likyoo/open-cd) change detection models.

[日本語版 README](README.ja.md)

## Why

[Open-CD](https://github.com/likyoo/open-cd) is a great change detection toolbox, but it depends on the OpenMMLab stack (`mmcv`, `mmengine`, `mmseg`, `mmpretrain`), which has become hard to install and to embed into other applications. **opencd-lite** re-implements Open-CD models as plain `nn.Module` classes so that:

- `pip install` is all you need — no `mim`, no source builds, no version-pin chains
- models can be imported and instantiated directly from any Python application
- **upstream Open-CD config files load unchanged**, and **published Open-CD checkpoints load without re-training**
- models export to ONNX so applications can run inference with `onnxruntime` alone

## Supported models

| Model | Paper | Open-CD `type` | Published checkpoint (LEVIR-CD) |
| --- | --- | --- | --- |
| CGNet | [Change Guiding Network (JSTARS 2023)](https://ieeexplore.ieee.org/document/10234560) | `CGNet` | [cgnet_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/cgnet_256x256_40k_levircd.pth) |
| IFN | [Deeply supervised image fusion network (ISPRS 2020)](https://www.sciencedirect.com/science/article/pii/S0924271620301532) | `IFN` | [ifn_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/ifn_256x256_40k_levircd.pth) |
| FC-EF | [Fully convolutional siamese networks (ICIP 2018)](https://ieeexplore.ieee.org/document/8451652) | `FC_EF` | [fc_ef_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/fc_ef_256x256_40k_levircd.pth) |
| FC-Siam-diff | [Fully convolutional siamese networks (ICIP 2018)](https://ieeexplore.ieee.org/document/8451652) | `FC_Siam_diff` | [fc_siam_diff_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/fc_siam_diff_256x256_40k_levircd.pth) |
| FC-Siam-conc | [Fully convolutional siamese networks (ICIP 2018)](https://ieeexplore.ieee.org/document/8451652) | `FC_Siam_conc` | [fc_siam_conc_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/fc_siam_conc_256x256_40k_levircd.pth) |
| SNUNet (ECAM) | [SNUNet-CD (GRSL 2021)](https://ieeexplore.ieee.org/document/9355573) | `SNUNet_ECAM` | [snunet_c16_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/snunet_c16_256x256_40k_levircd.pth) |

Checkpoints come from the official [Open-CD Model Zoo](https://huggingface.co/likyoo/Open-CD_Model_Zoo) on Hugging Face and load as-is (see `load_opencd_checkpoint`). More models will be ported incrementally.

## Installation

```bash
pip install opencd-lite            # core: torch, torchvision, numpy
pip install "opencd-lite[export]"  # + onnx, onnxruntime
pip install "opencd-lite[train]"   # + lightning, mlflow, pillow
```

Requires Python 3.11+.

## Quick start

### Direct instantiation — no config needed

```python
import torch
from opencd_lite import CGNet

model = CGNet(pretrained=False)  # plain nn.Module
x1 = torch.randn(1, 3, 256, 256)  # "before" image (normalized)
x2 = torch.randn(1, 3, 256, 256)  # "after" image (normalized)
change_map, final_map = model(x1, x2)
```

### From an Open-CD config + checkpoint

```python
from opencd_lite import build_model

detector = build_model(
    "configs/cgnet/cgnet_256x256_40k_levircd.py",  # upstream configs work as-is
    checkpoint="cgnet_levircd.pth",                # published Open-CD weights
)

# predict() reproduces the Open-CD test-time protocol
# (normalization, padding, whole/sliding-window inference, binarization)
import numpy as np
from PIL import Image

before = np.asarray(Image.open("before.png").convert("RGB"))
after = np.asarray(Image.open("after.png").convert("RGB"))
mask = detector.predict(before, after)  # (H, W) uint8, 1 = changed
```

### ONNX export

```python
from opencd_lite import export_onnx

export_onnx(detector, "cgnet.onnx", input_size=(256, 256))
```

The exported model needs only `onnxruntime` and `numpy` at deployment time. Inputs must be normalized as described in [`src/opencd_lite/transforms.py`](src/opencd_lite/transforms.py): RGB order, 0–255 scale, ImageNet mean/std.

### Training (with optional MLflow tracking)

```bash
pip install "opencd-lite[train]"
python tools/train.py configs/cgnet/cgnet_256x256_40k_levircd.py \
    --data-root /data/LEVIR-CD \
    --mlflow-uri http://localhost:5000   # omit to log to local CSV instead
```

Datasets are expected in the LEVIR-CD folder layout (`train/A`, `train/B`, `train/label`, …).

## Development

Everything runs inside Docker (CPU-only PyTorch):

```bash
docker compose build dev
docker compose run --rm dev            # run the test suite
docker compose run --rm dev ruff check .
```

An MLflow tracking server for training experiments is available behind a compose profile (it never starts by default):

```bash
docker compose --profile train up -d mlflow   # UI at http://localhost:5000
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards and the pull request workflow.

## Design rules

1. Model classes are plain `nn.Module`s depending only on `torch`/`torchvision` — no training-harness imports in `opencd_lite.models`.
2. Preprocessing is a fixed **specification** ([`transforms.py`](src/opencd_lite/transforms.py)), not configuration — checkpoint portability depends on it.
3. Configs describe *experiments*, not model definitions: every model is instantiable without any config machinery.
4. Module attribute names match upstream Open-CD so checkpoints load with nothing more than a `backbone.` prefix strip.

## License

[Apache License 2.0](LICENSE), the same license as Open-CD.

This project contains code derived from [Open-CD](https://github.com/likyoo/open-cd) (Copyright the Open-CD contributors), which in turn credits the original model authors — see the module docstrings for per-model attribution and paper references.
