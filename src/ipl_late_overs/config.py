"""Load config.yaml. Every path, season, split, threshold and seed comes from here."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ENV_VAR = "IPL_LATE_OVERS_CONFIG"


def config_path() -> Path:
    """Path of the active config file (override with $IPL_LATE_OVERS_CONFIG, e.g. in tests)."""
    return Path(os.environ.get(CONFIG_ENV_VAR, REPO_ROOT / "config.yaml"))


@lru_cache(maxsize=None)
def _load(path: Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg


def load_config() -> dict[str, Any]:
    return _load(config_path())


def config_hash() -> str:
    """SHA-256 of the config file bytes, stored alongside derived artefacts."""
    return hashlib.sha256(config_path().read_bytes()).hexdigest()


def path(key: str) -> Path:
    """Absolute path for a key in the `paths` section."""
    rel = load_config()["paths"][key]
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def _validate(cfg: dict[str, Any]) -> None:
    for section in ("paths", "download", "match", "late_overs", "history", "features", "splits", "seed"):
        assert section in cfg, f"config.yaml is missing section '{section}'"

    lo = cfg["late_overs"]
    assert 0 <= lo["first_over_index"] <= lo["last_over_index"] < cfg["match"]["overs"], lo

    s = cfg["splits"]
    train, val, test = s["train_seasons"], s["validation_seasons"], s["test_seasons"]
    assert train and val and test, "every split needs at least one season"
    all_seasons = train + val + test
    assert len(all_seasons) == len(set(all_seasons)), "a season appears in more than one split"
    assert max(train) < min(val) and max(val) < min(test), "splits must be chronological"
    assert val == sorted(val) and test == sorted(test), "season lists must be sorted"
