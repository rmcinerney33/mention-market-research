"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from mention_market.config import repo_root


@pytest.fixture(scope="session")
def project_root() -> Path:
    return repo_root()
