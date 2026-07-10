# opencd-lite へのコントリビューション

コントリビューションを検討いただきありがとうございます。このドキュメントでは開発環境の構築、コーディング規約、Pull Request の手順を説明します。

[English version](CONTRIBUTING.md)

## 開発環境

開発・テストはすべて Docker（CPU 版 PyTorch）内で行います:

```bash
docker compose build dev
docker compose run --rm dev                 # テストスイート全体を実行
docker compose run --rm dev pytest tests/test_models.py -v
docker compose run --rm dev ruff format .   # 自動フォーマット
docker compose run --rm dev ruff check .    # lint
docker compose run --rm dev mypy            # 型チェック
```

リポジトリはコンテナにバインドマウントされるため、ソースの編集は即座に反映されます。リビルドが必要なのは依存関係を変更したときだけです。

ローカルの virtualenv を使う場合:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -e ".[dev]"
```

## コーディング規約

- **言語**: コードコメント・docstring・ドキュメントはすべて**英語**で記述します。Markdown ドキュメントは `.ja.md` サフィックスの日本語訳を併置します（例: `README.md` → `README.ja.md`）。正本は英語版です。
- **フォーマット / lint**: `ruff format` と `ruff check` に合格すること（行長 100）。
- **型**: 公開関数には型ヒントを付け、`mypy` に合格すること。
- **テスト**: 新機能は pytest のテストとセットで提出します。テストは CPU のみで動作し、学習済み重みをダウンロードしてはいけません（モデルは `pretrained=False` で構築します）。

### アーキテクチャルール

1. `opencd_lite/models/` は素の `nn.Module` のみを置き、import は `torch` / `torchvision` に限定します。学習コード（Lightning、MLflow）は `opencd_lite/tasks/` に置き、models へ依存してよいが逆方向は禁止です。
2. 前処理の定数はモデルの**仕様**の一部であり、`transforms.py` に置きます。
3. モデルの属性名は本家 Open-CD と一致させ、公開 checkpoint が `backbone.` 接頭辞の除去だけでロードできるようにします（`checkpoint.py` 参照）。モデルクラス内で本家の命名を「綺麗に直す」ことはしないでください。
4. すべてのモデルは config なしで構築できること: `CGNet(pretrained=False)` がそのまま動くこと。

## Open-CD からの新モデル移植手順

1. 本家の `opencd/models/backbones/<name>.py` にモデルが自己完結しているか確認します（decode head が `IdentityHead` のモデルは移植が容易です）。
2. `src/opencd_lite/models/<name>.py` に属性名を同一に保って再実装し、`@register_model("<OpenCDTypeName>")` で登録します。
3. 関連する config を無変更で `configs/` にコピーします。
4. テストを追加します: forward 形状、config からの構築、checkpoint のラウンドトリップ、ONNX エクスポート。
5. `README.md` と `README.ja.md` の対応モデル表を更新します。
6. 参照した原著者実装のライセンスを確認し、モジュールの docstring に帰属を記載します。

## Pull Request の手順

1. フォークして `main` からトピックブランチを作成します。
2. 変更を加え、コンテナ内で `ruff format --check .`、`ruff check .`、`mypy`、`pytest` がすべて合格することを確認します。
3. コミットメッセージは英語（命令形。例: "Add SNUNet model"）で記述します。
4. 動機と変更内容を記載した Pull Request を作成します。レビュー前に CI がグリーンであることが必要です。

## Issue の報告

Issue テンプレートを使用してください。バグの場合は Python / PyTorch のバージョン、最小の再現手順、完全なトレースバックを含めてください。
