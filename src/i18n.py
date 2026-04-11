from __future__ import annotations

import os
from pathlib import Path


LANG = os.getenv("LLMS_LANG", "pt").strip().lower()


def tr(pt: str, en: str) -> str:
    return en if LANG.startswith("en") else pt


def figures_dir(root: Path) -> Path:
    path = Path(os.getenv("LLMS_FIGURES_DIR", str(root / "figures")))
    path.mkdir(parents=True, exist_ok=True)
    return path


def tables_dir(root: Path) -> Path:
    path = Path(os.getenv("LLMS_TABLES_DIR", str(root / "tables")))
    path.mkdir(parents=True, exist_ok=True)
    return path
