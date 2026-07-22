# CLAUDE.md — instructions for LLM coding assistants

This file gives coding agents the minimum context to work on opencd-lite.
The canonical documentation is [README.md](README.md) (project overview,
usage) and [CONTRIBUTING.md](CONTRIBUTING.md) (coding standards,
architecture rules, model-porting checklist) — read those first and keep
them, not this file, as the source of truth.

## What this project is

A lightweight, mmlab-free reimplementation of
[Open-CD](https://github.com/likyoo/open-cd) change detection models.
Upstream Open-CD configs and published checkpoints must keep working —
that compatibility is the core product constraint.

## Hard rules

- All code comments, docstrings, and docs are in **English**. Every
  Markdown doc has a Japanese sibling with the `.ja.md` suffix; update
  both when you change one (English is canonical).
- `src/opencd_lite/models/` may import only `torch`/`torchvision` —
  never Lightning, MLflow, or anything from `tasks/`.
- Do not rename model module attributes: they mirror upstream Open-CD
  checkpoint keys (`backbone.<attr>...`). Renaming breaks weight loading.
- Preprocessing constants live in `src/opencd_lite/transforms.py` as a
  fixed specification. Do not duplicate or "configure" them elsewhere.
- Tests must run CPU-only and must not download pretrained weights
  (always construct models with `pretrained=False` in tests).

## Commands (run inside the dev container)

```bash
docker compose build dev              # once, or after dependency changes
docker compose run --rm dev           # pytest (full suite)
docker compose run --rm dev ruff format .
docker compose run --rm dev ruff check .
docker compose run --rm dev mypy
```

CI (`.github/workflows/ci.yml`) enforces: `ruff format --check`,
`ruff check`, `mypy`, and `pytest` (including ONNX-export and training
smoke tests) on Python 3.11 and 3.12.

## Layout

```
src/opencd_lite/
  models/       plain nn.Module models + registry (torch/torchvision only)
  config.py     mmengine-style .py config loader (no mmengine)
  builder.py    Open-CD config -> ChangeDetector
  checkpoint.py Open-CD checkpoint loading (backbone.* prefix strip)
  transforms.py preprocessing specification (fixed constants); torch-free
                constants + numpy helpers, torch imported lazily
  protocol.py   InferenceConfig (test-time protocol dataclass, torch-free)
  inference.py  ChangeDetector: whole/slide inference, binarization
  export.py     ONNX export (+ verification, embeds InferenceConfig metadata)
  onnx.py       ONNXChangeDetector: torch-free inference (numpy + onnxruntime)
  datasets/     LEVIR-CD-layout folder dataset (train extra)
  tasks/        Lightning training task (train extra)
  data_prep/    dataset preparation utilities (dataprep extra: alignment, cropping, splitting)
configs/        Open-CD configs, copied unchanged from upstream
tools/train.py  training entry point (optional MLflow tracking)
tools/export.py ONNX export CLI
tools/infer_onnx.py  torch-free ONNX inference CLI
tools/prepare_data.py  dataset preparation CLI
tests/          pytest suite (markers: export, onnx, train, dataprep)
```

The core install is torch-free (`numpy` only); `torch`/`torchvision` are
the `[torch]` extra. `import opencd_lite` is lazy (PEP 562) so the ONNX
inference path (`opencd_lite.onnx`, `opencd_lite.protocol`, the torch-free
parts of `transforms.py`) works without PyTorch installed.

When porting a new model, follow the checklist in
[CONTRIBUTING.md](CONTRIBUTING.md#porting-a-new-model-from-open-cd).
