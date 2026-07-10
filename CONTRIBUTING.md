# Contributing to opencd-lite

Thank you for considering a contribution! This document describes how to set up a development environment, the coding standards, and the pull request workflow.

[日本語版](CONTRIBUTING.ja.md)

## Development environment

All development and testing happens inside Docker with CPU-only PyTorch:

```bash
docker compose build dev
docker compose run --rm dev                 # run the full test suite
docker compose run --rm dev pytest tests/test_models.py -v
docker compose run --rm dev ruff format .   # auto-format
docker compose run --rm dev ruff check .    # lint
docker compose run --rm dev mypy            # type check
```

The repository is bind-mounted into the container, so source edits take effect immediately; rebuild only when dependencies change.

If you prefer a local virtualenv:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -e ".[dev]"
```

## Coding standards

- **Language**: all code comments, docstrings, and documentation are written in **English**. Markdown documents additionally get a Japanese translation with the `.ja.md` suffix (e.g. `README.md` → `README.ja.md`); the English version is canonical.
- **Formatting / linting**: `ruff format` and `ruff check` must pass (line length 100).
- **Typing**: public functions carry type hints; `mypy` must pass.
- **Tests**: new functionality ships together with pytest tests. Tests run CPU-only and must not download pretrained weights (construct models with `pretrained=False`).

### Architecture rules

1. `opencd_lite/models/` contains plain `nn.Module`s and may import only `torch`/`torchvision`. Training code (Lightning, MLflow) lives in `opencd_lite/tasks/` and may depend on models — never the reverse.
2. Preprocessing constants are part of the model **specification** and live in `transforms.py`.
3. Model attribute names must match upstream Open-CD so published checkpoints load via a `backbone.` prefix strip (see `checkpoint.py`). Do not "clean up" upstream naming inside model classes.
4. Every model must be constructible without a config: `CGNet(pretrained=False)` just works.

## Porting a new model from Open-CD

1. Check that the upstream model is self-contained in `opencd/models/backbones/<name>.py` (models whose decode heads are `IdentityHead` port most easily).
2. Re-implement it under `src/opencd_lite/models/<name>.py`, keeping attribute names identical, and register it with `@register_model("<OpenCDTypeName>")`.
3. Copy the relevant configs into `configs/` unchanged.
4. Add tests: forward shapes, config build, checkpoint round-trip, ONNX export.
5. Update the supported model tables in `README.md` and `README.ja.md`.
6. Verify the license of any referenced original-author implementation and attribute it in the module docstring.

## Pull request workflow

1. Fork and create a topic branch from `main`.
2. Make your changes; ensure `ruff format --check .`, `ruff check .`, `mypy` and `pytest` all pass in the container.
3. Write commit messages in English (imperative mood, e.g. "Add SNUNet model").
4. Open a pull request describing the motivation and the changes. CI must be green before review.

## Reporting issues

Use the issue templates. For bugs, include the Python/PyTorch versions, a minimal reproduction, and the full traceback.
