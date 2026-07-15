"""Env-backed config. Reads <repo>/.env once; real environment variables win.

Never print or log secret values.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _REPO_ROOT / ".env"


def _load_env() -> None:
    if not _ENV_PATH.exists():
        return
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()


def require(name: str) -> str:
    value = os.environ.get(name, "")
    if not value or value.startswith("<") or value in ("sk-xxxx", "changeme"):
        raise RuntimeError(f"Missing or placeholder env var: {name}")
    return value


QWEN_BASE_URL = os.environ.get(
    "QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

# Heterogeneous backbones — the one debate lever with robust evidence
# (arXiv:2502.08788 "universal antidote"). Overridable per-role via env.
MODEL_STRONG = os.environ.get("MAJALIS_MODEL_STRONG", "qwen3.7-max")
MODEL_MID = os.environ.get("MAJALIS_MODEL_MID", os.environ.get("QWEN_MODEL", "qwen3.7-plus"))
MODEL_FAST = os.environ.get("MAJALIS_MODEL_FAST", "qwen3.6-flash")

DB_PATH = str(_REPO_ROOT / "data" / "majalis_beliefs.db")
