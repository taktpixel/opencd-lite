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
