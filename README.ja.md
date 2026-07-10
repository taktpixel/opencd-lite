# opencd-lite

[Open-CD](https://github.com/likyoo/open-cd) の変化検出モデルを、mmlab 非依存の軽量な PyTorch 実装として提供します。

[English README](README.md)

## なぜ作ったか

[Open-CD](https://github.com/likyoo/open-cd) は優れた変化検出ツールボックスですが、OpenMMLab スタック（`mmcv`、`mmengine`、`mmseg`、`mmpretrain`）に依存しており、インストールや他アプリケーションへの組み込みが困難になっています。**opencd-lite** は Open-CD のモデルを素の `nn.Module` として再実装し、以下を実現します:

- `pip install` だけで環境構築が完結（`mim` 不要、ソースビルド不要、バージョン固定の連鎖なし）
- 任意の Python アプリケーションから直接 import・インスタンス化できる
- **本家 Open-CD の config ファイルをそのまま読み込める**。**公開済みの学習済み重みを再学習なしで流用できる**
- ONNX にエクスポートでき、アプリ側は `onnxruntime` だけで推論できる

## 対応モデル

| モデル | 論文 | Open-CD `type` | 公開 checkpoint（LEVIR-CD） |
| --- | --- | --- | --- |
| CGNet | [Change Guiding Network (JSTARS 2023)](https://ieeexplore.ieee.org/document/10234560) | `CGNet` | [cgnet_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/cgnet_256x256_40k_levircd.pth) |
| IFN | [Deeply supervised image fusion network (ISPRS 2020)](https://www.sciencedirect.com/science/article/pii/S0924271620301532) | `IFN` | [ifn_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/ifn_256x256_40k_levircd.pth) |
| FC-EF | [Fully convolutional siamese networks (ICIP 2018)](https://ieeexplore.ieee.org/document/8451652) | `FC_EF` | [fc_ef_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/fc_ef_256x256_40k_levircd.pth) |
| FC-Siam-diff | [Fully convolutional siamese networks (ICIP 2018)](https://ieeexplore.ieee.org/document/8451652) | `FC_Siam_diff` | [fc_siam_diff_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/fc_siam_diff_256x256_40k_levircd.pth) |
| FC-Siam-conc | [Fully convolutional siamese networks (ICIP 2018)](https://ieeexplore.ieee.org/document/8451652) | `FC_Siam_conc` | [fc_siam_conc_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/fc_siam_conc_256x256_40k_levircd.pth) |
| SNUNet (ECAM) | [SNUNet-CD (GRSL 2021)](https://ieeexplore.ieee.org/document/9355573) | `SNUNet_ECAM` | [snunet_c16_256x256_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/snunet_c16_256x256_40k_levircd.pth) |

checkpoint は Hugging Face 上の公式 [Open-CD Model Zoo](https://huggingface.co/likyoo/Open-CD_Model_Zoo) のものをそのままロードできます（`load_opencd_checkpoint` 参照）。今後インクリメンタルに追加していきます。

## インストール

```bash
pip install opencd-lite            # コア: torch, torchvision, numpy
pip install "opencd-lite[export]"  # + onnx, onnxruntime
pip install "opencd-lite[train]"   # + lightning, mlflow, pillow
```

Python 3.11 以上が必要です。

## クイックスタート

### 直接インスタンス化 — config 不要

```python
import torch
from opencd_lite import CGNet

model = CGNet(pretrained=False)  # 素の nn.Module
x1 = torch.randn(1, 3, 256, 256)  # 変化前画像（正規化済み）
x2 = torch.randn(1, 3, 256, 256)  # 変化後画像（正規化済み）
change_map, final_map = model(x1, x2)
```

### Open-CD config + checkpoint から構築

```python
from opencd_lite import build_model

detector = build_model(
    "configs/cgnet/cgnet_256x256_40k_levircd.py",  # 本家 config がそのまま使える
    checkpoint="cgnet_levircd.pth",                # 公開済み Open-CD 重み
)

# predict() は Open-CD のテスト時プロトコルを再現します
# （正規化、パディング、whole / スライディングウィンドウ推論、二値化）
import numpy as np
from PIL import Image

before = np.asarray(Image.open("before.png").convert("RGB"))
after = np.asarray(Image.open("after.png").convert("RGB"))
mask = detector.predict(before, after)  # (H, W) uint8、1 = 変化あり
```

### ONNX エクスポート

```python
from opencd_lite import export_onnx

export_onnx(detector, "cgnet.onnx", input_size=(256, 256))
```

エクスポートしたモデルの実行に必要なのは `onnxruntime` と `numpy` のみです。入力は [`src/opencd_lite/transforms.py`](src/opencd_lite/transforms.py) に記載の仕様（RGB 順、0–255 スケール、ImageNet mean/std）で正規化してください。

### 学習（MLflow 記録はオプション）

```bash
pip install "opencd-lite[train]"
python tools/train.py configs/cgnet/cgnet_256x256_40k_levircd.py \
    --data-root /data/LEVIR-CD \
    --mlflow-uri http://localhost:5000   # 省略時はローカル CSV に記録
```

データセットは LEVIR-CD のフォルダ構成（`train/A`、`train/B`、`train/label` など）を想定しています。

## 開発

開発・テストはすべて Docker（CPU 版 PyTorch）内で行います:

```bash
docker compose build dev
docker compose run --rm dev            # テストスイートを実行
docker compose run --rm dev ruff check .
```

学習実験用の MLflow トラッキングサーバは compose プロファイルで分離されています（既定では起動しません）:

```bash
docker compose --profile train up -d mlflow   # UI: http://localhost:5000
```

コーディング規約と Pull Request の手順は [CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。

## 設計ルール

1. モデルクラスは `torch` / `torchvision` のみに依存する素の `nn.Module` — `opencd_lite.models` に学習ハーネスの import を持ち込まない
2. 前処理は設定ではなく固定の**仕様**（[`transforms.py`](src/opencd_lite/transforms.py)）— 重みの移植性はここで決まる
3. config は「実験の記述」であって「モデル定義の記述」ではない — すべてのモデルは config なしでインスタンス化できる
4. モジュールの属性名は本家 Open-CD と一致させ、checkpoint を `backbone.` 接頭辞の除去だけでロードできるようにする

## ライセンス

[Apache License 2.0](LICENSE)（Open-CD と同一）。

本プロジェクトは [Open-CD](https://github.com/likyoo/open-cd)（Copyright the Open-CD contributors）由来のコードを含みます。モデルごとの原著者クレジット・論文情報は各モジュールの docstring を参照してください。
