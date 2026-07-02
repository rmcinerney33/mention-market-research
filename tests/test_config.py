"""Smoke tests for the config loader.

Focused on invariants the rest of the pipeline will rely on:
- config.yaml exists and parses.
- Required top-level keys are present so downstream callers can trust the shape.
- Local overrides merge correctly (deep merge, not shallow replace).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from mention_market.config import load_config

REQUIRED_TOP_LEVEL_KEYS = {
    "project",
    "sources",
    "paths",
    "targets",
    "date_ranges",
    "market_snapshots",
    "features",
    "modeling",
}


def test_default_config_loads(project_root: Path) -> None:
    cfg = load_config(local_override=None)
    missing = REQUIRED_TOP_LEVEL_KEYS - cfg.keys()
    assert not missing, f"config.yaml missing required top-level keys: {missing}"


def test_default_config_paths_are_relative(project_root: Path) -> None:
    cfg = load_config(local_override=None)
    for key, value in cfg["paths"].items():
        assert not Path(value).is_absolute(), (
            f"paths.{key} must be repo-relative, got absolute path: {value}"
        )


def test_lead_times_are_positive_and_sorted_desc(project_root: Path) -> None:
    cfg = load_config(local_override=None)
    lead_times = cfg["market_snapshots"]["lead_times_hours"]
    assert all(t > 0 for t in lead_times), "lead times must be positive hours-before-event"
    assert lead_times == sorted(lead_times, reverse=True), (
        "lead times should be sorted descending (farthest-out first) for readability"
    )


def test_local_override_deep_merges(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    override_path = tmp_path / "override.yaml"
    base_path.write_text(
        textwrap.dedent(
            """
            project:
              random_seed: 42
              timezone: "America/New_York"
            paths:
              raw_dir: "data/raw"
            """
        ).strip()
    )
    override_path.write_text(
        textwrap.dedent(
            """
            project:
              random_seed: 7
            """
        ).strip()
    )

    merged = load_config(path=base_path, local_override=override_path)
    # Overridden value wins, untouched sibling key survives, unrelated section untouched.
    assert merged["project"]["random_seed"] == 7
    assert merged["project"]["timezone"] == "America/New_York"
    assert merged["paths"]["raw_dir"] == "data/raw"


def test_local_override_missing_is_ok(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text("project:\n  random_seed: 1\n")
    cfg = load_config(path=base_path, local_override=tmp_path / "does-not-exist.yaml")
    assert cfg["project"]["random_seed"] == 1


@pytest.mark.parametrize(
    "yaml_text,expected",
    [
        ("", {}),
        ("project: null", {"project": None}),
    ],
)
def test_empty_and_null_yaml(tmp_path: Path, yaml_text: str, expected: dict) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml_text)
    cfg = load_config(path=base_path, local_override=None)
    assert cfg == expected


def test_config_yaml_parses_as_valid_yaml(project_root: Path) -> None:
    """Independent of load_config: the shipped config file is legal YAML."""
    with open(project_root / "config.yaml") as f:
        parsed = yaml.safe_load(f)
    assert isinstance(parsed, dict)
