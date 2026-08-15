# opencd-lite

Lightweight, mmlab-free PyTorch implementation of [Open-CD](https://github.com/likyoo/open-cd) change detection models.

[日本語版 README](README.ja.md)

## Why

[Open-CD](https://github.com/likyoo/open-cd) is a great change detection toolbox, but it depends on the OpenMMLab stack (`mmcv`, `mmengine`, `mmseg`, `mmpretrain`), which has become hard to install and to embed into other applications. **opencd-lite** re-implements Open-CD models as plain `nn.Module` classes so that:

- `pip install` is all you need — no `mim`, no source builds, no version-pin chains
- models can be imported and instantiated directly from any Python application
- **upstream Open-CD config files load unchanged**, and **published Open-CD checkpoints load without re-training**
- models export to ONNX and run inference with `onnxruntime` alone — the core install is torch-free, so a deployment can skip PyTorch entirely

## Supported models

| Model | Paper | Open-CD `type` | Published checkpoint (LEVIR-CD) |
| --- | --- | --- | --- |
| CGNet | [Change Guiding Network (JSTARS 2023)](https://ieeexplore.ieee.org/document/10234560) | `CGNet` | [cgnet_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/cgnet_256x256_40k_levircd.pth) |
| IFN | [Deeply supervised image fusion network (ISPRS 2020)](https://www.sciencedirect.com/science/article/pii/S0924271620301532) | `IFN` | [ifn_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/ifn_256x256_40k_levircd.pth) |
| FC-EF | [Fully convolutional siamese networks (ICIP 2018)](https://ieeexplore.ieee.org/document/8451652) | `FC_EF` | [fc_ef_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/fc_ef_256x256_40k_levircd.pth) |
| FC-Siam-diff | [Fully convolutional siamese networks (ICIP 2018)](https://ieeexplore.ieee.org/document/8451652) | `FC_Siam_diff` | [fc_siam_diff_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/fc_siam_diff_256x256_40k_levircd.pth) |
| FC-Siam-conc | [Fully convolutional siamese networks (ICIP 2018)](https://ieeexplore.ieee.org/document/8451652) | `FC_Siam_conc` | [fc_siam_conc_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/fc_siam_conc_256x256_40k_levircd.pth) |
| SNUNet (ECAM) | [SNUNet-CD (GRSL 2021)](https://ieeexplore.ieee.org/document/9355573) | `SNUNet_ECAM` | [snunet_c16_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/snunet_c16_256x256_40k_levircd.pth) |
| BIT | [Remote Sensing Image Change Detection with Transformers (TGRS 2022)](https://ieeexplore.ieee.org/document/9491802) | `mmseg.ResNetV1c` + `BITHead` | [bit_r18_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/bit_r18_256x256_40k_levircd.pth) |
| Changer (ChangerEx) | [Changer: Feature Interaction Is What You Need (TGRS 2023)](https://ieeexplore.ieee.org/document/10123098) | `IA_ResNetV1c` + `Changer` | [ChangerEx_r18-512x512_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/ChangerEx_r18-512x512_40k_levircd.pth) |
| STANet (PAM) | [A Spatial-Temporal Attention-Based Method (RS 2020)](https://www.mdpi.com/2072-4292/12/10/1662) | `mmseg.ResNetV1c` + `STAHead` | [stanet_pam_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/stanet_pam_256x256_40k_levircd.pth) |
| LightCDNet (base/large) | [LightCDNet: Lightweight Change Detection Network (GRSL 2023)](https://ieeexplore.ieee.org/document/10214556) | `LightCDNet` + `DS_FPNHead` | [lightcdnet_b_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/lightcdnet_b_256x256_40k_levircd.pth), [lightcdnet_l_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/lightcdnet_l_256x256_40k_levircd.pth) |
| ChangeFormer (MiT-b0/b1) | [A Transformer-Based Siamese Network for Change Detection (IGARSS 2022)](https://ieeexplore.ieee.org/document/9883686) | `mmseg.MixVisionTransformer` + `mmseg.SegformerHead` | [changeformer_mit-b0_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/changeformer_mit-b0_256x256_40k_levircd.pth), [changeformer_mit-b1_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/changeformer_mit-b1_256x256_40k_levircd.pth) |
| ChangeStar (FarSeg) | [Change is Everywhere (ICCV 2021)](https://arxiv.org/abs/2108.07002) | `FarSegFPN` + `ChangeStarHead` | [changestar_farseg_1x96_512x512_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/changestar_farseg_1x96_512x512_40k_levircd.pth) |
| TinyCD v2 (S/B/L) | [TinyCD v2 (Open-CD technical report)](https://arxiv.org/abs/2407.15317) | `TinyNet` + `TinyHead` | — (no published checkpoint; verified architecturally against upstream) |

Checkpoints come from the official [Open-CD Model Zoo](https://huggingface.co/likyoo/Open-CD_Model_Zoo) on Hugging Face and load as-is (see `load_opencd_checkpoint`). More models will be ported incrementally.

## Installation

The **core install is torch-free** — enough to run inference from an
exported ONNX graph. The PyTorch models, training and ONNX *export* live
behind extras that pull `torch` in, so a deployment that only runs
onnxruntime never installs torch.

```bash
pip install opencd-lite              # core: numpy only
pip install "opencd-lite[onnx]"      # torch-free inference: + onnxruntime, pillow
pip install "opencd-lite[torch]"     # PyTorch models: + torch, torchvision
pip install "opencd-lite[export]"    # export models to ONNX: + torch, onnx, onnxruntime
pip install "opencd-lite[train]"     # + lightning, mlflow, pillow
pip install "opencd-lite[dataprep]"  # + opencv-python-headless, pillow
```

Requires Python 3.11+. The direct-instantiation and `build_model` APIs
below need the `torch` (or `train`/`export`) extra; the ONNX inference
section needs only `onnx`.

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
    checkpoint="cgnet_levircd.pth",  # published Open-CD weights
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

The export embeds the model's test-time protocol (whole/slide, crop,
stride, threshold) into the ONNX metadata, so the graph is
self-describing.

### Torch-free ONNX inference

`ONNXChangeDetector` reproduces the full Open-CD test-time protocol —
normalization, padding, whole/sliding-window inference and
binarization — on the exported graph, using only `numpy` and
`onnxruntime`. No PyTorch is needed at deployment time. The protocol is
implemented op-for-op against the PyTorch `ChangeDetector`, and its masks
match it exactly on every input tested (30/30 real 256×256 patches and a
2927×3197 slide pair on a trained CGNet checkpoint).

```python
import numpy as np
from PIL import Image
from opencd_lite.onnx import ONNXChangeDetector

# The inference protocol is read from the ONNX metadata written at export.
detector = ONNXChangeDetector.from_file("cgnet.onnx")

before = np.asarray(Image.open("before.png").convert("RGB"))
after = np.asarray(Image.open("after.png").convert("RGB"))
mask = detector.predict(before, after)  # (H, W) uint8, 1 = changed
```

The exported graph is fixed-size. **Slide mode** (the default for these
configs) tiles the image into windows of the exported size, so it handles
inputs of any size at or above the crop size. **Whole mode** expects
inputs of exactly the exported size (e.g. pre-tiled 256×256 patches). A
mismatched size raises a clear error rather than running incorrectly.
Inputs are normalized internally per
[`src/opencd_lite/transforms.py`](src/opencd_lite/transforms.py): RGB
order, 0–255 scale, ImageNet mean/std.

Command-line, torch-free end to end:

```bash
pip install "opencd-lite[export]"    # to write the graph (needs torch once)
python tools/export.py configs/cgnet/cgnet_256x256_40k_levircd.py \
    --checkpoint cgnet_levircd.pth -o cgnet.onnx --input-size 256 256

pip install "opencd-lite[onnx]"      # to run it (no torch)
python tools/infer_onnx.py cgnet.onnx before.png after.png -o mask.png --scale
```

### Dataset preparation

Turn raw before/after image folders into a LEVIR-CD-layout dataset that
`tools/train.py` can consume. The `dataprep` extra adds a `tools/prepare_data.py`
CLI with six subcommands:

```bash
pip install "opencd-lite[dataprep]"

# Align after-images onto before-images (keypoint matching)
python tools/prepare_data.py align before/ after/ -o aligned/ --method sift

# Random crops + balanced train/val/test split in the LEVIR layout
python tools/prepare_data.py crop \
    --image-from before/ --image-to aligned/ --label label/ \
    -o dataset/ --crop-size 256 --count 30 --split 0.6 0.2 0.2

# Shuffle & split parallel directories into train/val
python tools/prepare_data.py split A/ B/ label/ -o dataset/ --ratio 0.8

# Keep only filenames common to all directories
python tools/prepare_data.py intersect A/ B/ label/ -o common/

# Convert images to PNG (mirrors the input subtree)
python tools/prepare_data.py convert raw/ -o png/ --pattern '*.bmp' --to png

# Sliding-window tiling
python tools/prepare_data.py tile images/ -o tiles/ \
    --tile-size 1024 1024 --stride 512 512
```

End to end, aligning `B` onto `A` and cropping into a training-ready dataset:

```bash
python tools/prepare_data.py align A/ B/ -o aligned/
python tools/prepare_data.py crop \
    --image-from A/ --image-to aligned/ --label label/ \
    -o dataset/ --split 0.6 0.2 0.2
# dataset/train/{A,B,label}, dataset/val/..., dataset/test/... ready for tools/train.py
```

See `python tools/prepare_data.py <command> --help` for the full option list.

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
