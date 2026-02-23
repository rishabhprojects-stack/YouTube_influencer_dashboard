from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else Path("configs/default.yaml")
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
