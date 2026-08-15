# opencd-lite

[Open-CD](https://github.com/likyoo/open-cd) の変化検出モデルを、mmlab 非依存の軽量な PyTorch 実装として提供します。

[English README](README.md)

## なぜ作ったか

[Open-CD](https://github.com/likyoo/open-cd) は優れた変化検出ツールボックスですが、OpenMMLab スタック（`mmcv`、`mmengine`、`mmseg`、`mmpretrain`）に依存しており、インストールや他アプリケーションへの組み込みが困難になっています。**opencd-lite** は Open-CD のモデルを素の `nn.Module` として再実装し、以下を実現します:

- `pip install` だけで環境構築が完結（`mim` 不要、ソースビルド不要、バージョン固定の連鎖なし）
- 任意の Python アプリケーションから直接 import・インスタンス化できる
- **本家 Open-CD の config ファイルをそのまま読み込める**。**公開済みの学習済み重みを再学習なしで流用できる**
- ONNX にエクスポートでき、アプリ側は `onnxruntime` だけで推論できる。コアインストールは torch 非依存なので、デプロイ側は PyTorch を一切入れずに済む

## 対応モデル

| モデル | 論文 | Open-CD `type` | 公開 checkpoint（LEVIR-CD） |
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
| TinyCD v2 (S/B/L) | [TinyCD v2 (Open-CD technical report)](https://arxiv.org/abs/2407.15317) | `TinyNet` + `TinyHead` | —（公開 checkpoint なし。本家実装とのアーキテクチャ一致検証済み） |
| BAN (ViT-L/14 CLIP + MiT-b0) | [A New Learning Paradigm for Foundation Model-based RS Change Detection (TGRS 2024)](https://arxiv.org/abs/2312.01163) | `mmseg.VisionTransformer` + `BitemporalAdapterHead` | [ban_vit-l14-clip_mit-b0_512x512_40k_levircd.pth](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/ban_vit-l14-clip_mit-b0_512x512_40k_levircd.pth) |

checkpoint は Hugging Face 上の公式 [Open-CD Model Zoo](https://huggingface.co/likyoo/Open-CD_Model_Zoo) のものをそのままロードできます（`load_opencd_checkpoint` 参照）。今後インクリメンタルに追加していきます。

## インストール

**コアインストールは torch 非依存**で、エクスポート済みの ONNX グラフからの推論だけならこれで完結します。PyTorch モデル・学習・ONNX への*エクスポート*は `torch` を引き込む extra の裏側にあるため、onnxruntime だけで動かすデプロイ環境に torch が入ることはありません。

```bash
pip install opencd-lite              # コア: numpy のみ
pip install "opencd-lite[onnx]"      # torch 非依存の推論: + onnxruntime, pillow
pip install "opencd-lite[torch]"     # PyTorch モデル: + torch, torchvision
pip install "opencd-lite[export]"    # ONNX へのエクスポート: + torch, onnx, onnxruntime
pip install "opencd-lite[train]"     # + lightning, mlflow, pillow
pip install "opencd-lite[dataprep]"  # + opencv-python-headless, pillow
```

Python 3.11 以上が必要です。以下の「直接インスタンス化」「build_model」の API には `torch`（または `train`/`export`）extra が必要です。ONNX 推論のセクションは `onnx` extra だけで動きます。

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
    checkpoint="cgnet_levircd.pth",  # 公開済み Open-CD 重み
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

エクスポート時に、モデルのテスト時プロトコル（whole/slide、crop、stride、threshold）が ONNX のメタデータへ埋め込まれるため、グラフ単体で推論設定を復元できます。

### torch 非依存の ONNX 推論

`ONNXChangeDetector` は Open-CD のテスト時プロトコル全体（正規化、パディング、whole / スライディングウィンドウ推論、二値化）を、エクスポート済みグラフの上で `numpy` と `onnxruntime` のみを使って再現します。デプロイ時に PyTorch は不要です。プロトコルは PyTorch 版 `ChangeDetector` と演算単位で対応するよう実装されており、これまで検証したすべての入力（学習済み CGNet checkpoint での実データ 256×256 パッチ 30/30、および 2927×3197 の slide 推論）で出力マスクが完全一致しています。

```python
import numpy as np
from PIL import Image
from opencd_lite.onnx import ONNXChangeDetector

# 推論プロトコルはエクスポート時に書き込まれた ONNX メタデータから読み込まれます。
detector = ONNXChangeDetector.from_file("cgnet.onnx")

before = np.asarray(Image.open("before.png").convert("RGB"))
after = np.asarray(Image.open("after.png").convert("RGB"))
mask = detector.predict(before, after)  # (H, W) uint8、1 = 変化あり
```

エクスポートされたグラフは固定サイズです。**slide モード**（これらの config の既定）は画像をエクスポートサイズのウィンドウに分割して処理するため、crop サイズ以上であれば任意の入力サイズを扱えます。**whole モード**はエクスポートサイズと完全に一致する入力（例: 256×256 に切り出し済みのパッチ）を想定します。サイズが一致しない場合は、誤った推論をせず明確なエラーを送出します。入力は [`src/opencd_lite/transforms.py`](src/opencd_lite/transforms.py) の仕様（RGB 順、0–255 スケール、ImageNet mean/std）で内部的に正規化されます。

コマンドラインで、エクスポートから推論まで torch なしで実行:

```bash
pip install "opencd-lite[export]"    # グラフ書き出し（この時だけ torch が必要）
python tools/export.py configs/cgnet/cgnet_256x256_40k_levircd.py \
    --checkpoint cgnet_levircd.pth -o cgnet.onnx --input-size 256 256

pip install "opencd-lite[onnx]"      # 実行（torch 不要）
python tools/infer_onnx.py cgnet.onnx before.png after.png -o mask.png --scale
```

### データセット準備

変化前 / 変化後の生画像フォルダから、`tools/train.py` がそのまま利用できる
LEVIR-CD フォルダ構成のデータセットを作成します。`dataprep` extra を入れると、
6 つのサブコマンドを持つ `tools/prepare_data.py` CLI が使えます:

```bash
pip install "opencd-lite[dataprep]"

# 変化後画像を変化前画像に位置合わせ（キーポイントマッチング）
python tools/prepare_data.py align before/ after/ -o aligned/ --method sift

# ランダムクロップ + バランス調整した train/val/test 分割（LEVIR 構成）
python tools/prepare_data.py crop \
    --image-from before/ --image-to aligned/ --label label/ \
    -o dataset/ --crop-size 256 --count 30 --split 0.6 0.2 0.2

# 並列ディレクトリをシャッフルして train/val に分割
python tools/prepare_data.py split A/ B/ label/ -o dataset/ --ratio 0.8

# 全ディレクトリに共通するファイル名だけを残す
python tools/prepare_data.py intersect A/ B/ label/ -o common/

# 画像を PNG に変換（入力のサブツリー構造を維持）
python tools/prepare_data.py convert raw/ -o png/ --pattern '*.bmp' --to png

# スライディングウィンドウによるタイリング
python tools/prepare_data.py tile images/ -o tiles/ \
    --tile-size 1024 1024 --stride 512 512
```

`B` を `A` に位置合わせしてから、学習可能なデータセットへクロップするまでの一連の例:

```bash
python tools/prepare_data.py align A/ B/ -o aligned/
python tools/prepare_data.py crop \
    --image-from A/ --image-to aligned/ --label label/ \
    -o dataset/ --split 0.6 0.2 0.2
# dataset/train/{A,B,label}、dataset/val/...、dataset/test/... が tools/train.py 用に生成される
```

各オプションの詳細は `python tools/prepare_data.py <command> --help` を参照してください。

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
