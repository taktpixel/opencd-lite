"""Shared fixtures for the opencd-lite test suite.

All tests run CPU-only and never download pretrained weights
(models are constructed with ``pretrained=False``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    return REPO_ROOT / "configs"


@pytest.fixture(autouse=True)
def _deterministic_seed() -> None:
    torch.manual_seed(0)


@pytest.fixture()
def make_small_ban_head():
    """Factory building a tiny BAN head (shared across BAN tests)."""

    def _build():
        from opencd_lite.models import BitemporalAdapterHead

        return BitemporalAdapterHead(
            ban_cfg={
                "clip_channels": 24,
                "fusion_index": [1],
                "side_enc_cfg": {
                    "type": "mmseg.MixVisionTransformer",
                    "embed_dims": 8,
                    "num_stages": 2,
                    "num_layers": [1, 1],
                    "num_heads": [1, 2],
                    "patch_sizes": [7, 3],
                    "strides": [4, 2],
                    "sr_ratios": [4, 2],
                    "out_indices": (0, 1),
                },
            },
            ban_dec_cfg={
                "type": "BAN_MLPDecoder",
                "in_channels": [8, 16],
                "channels": 8,
                "num_classes": 2,
                "dropout_ratio": 0.0,
            },
        )

    return _build


@pytest.fixture(scope="session")
def cgnet_small():
    """A CGNet without pretrained weights, shared across tests (read-only)."""
    from opencd_lite import CGNet

    model = CGNet(pretrained=False)
    model.eval()
    return model


@pytest.fixture(scope="session")
def ifn_small():
    """An IFN without pretrained weights, shared across tests (read-only)."""
    from opencd_lite import IFN

    model = IFN(pretrained=False)
    model.eval()
    return model
