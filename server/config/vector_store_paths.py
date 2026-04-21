"""Resolve vector store paths from vector_store.json relative to project root."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).resolve().parent / "vector_store.json"


def load_vector_store_paths(project_root: Path) -> tuple[Path, Path]:
    """
    Return (index_path, doc_path) as absolute paths under project_root.

    Reads ``index_path`` and ``doc_path`` from server/config/vector_store.json;
    defaults to data/index and data/docs if the file or keys are missing.
    """
    data: dict[str, Any] = {}
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
    idx = data.get("index_path") or "data/index"
    doc = data.get("doc_path") or "data/docs"
    if not isinstance(idx, str) or not isinstance(doc, str):
        idx, doc = "data/index", "data/docs"
    return (project_root / idx).resolve(), (project_root / doc).resolve()
