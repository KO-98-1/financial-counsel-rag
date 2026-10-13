"""config.yaml 로드와 프로젝트 경로 해석."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(rel: str | Path) -> Path:
    """프로젝트 루트 기준 상대경로를 절대경로로."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p
