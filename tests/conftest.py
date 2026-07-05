"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the src layout is importable even if the editable install's .pth gets
# clobbered by an unrelated `pip install` (a recurring fragility in this repo).
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest  # noqa: E402

from mention_market.config import repo_root  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return repo_root()
