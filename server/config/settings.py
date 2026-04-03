"""Load embedding-related settings from the packaged JSON config."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).resolve().parent / "embedding.json"


@dataclass(frozen=True)
class EmbeddingSettings:
    """Resolved embedding client parameters for one model type."""

    base_url: str
    api_key: str
    embedding_dim: int
    prefix: str
    embedding_model: str | None


def _load_raw_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or _CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def _detect_model_type() -> str:
    import urllib.request
    for model_type, host in [("ollama_local", "http://localhost:11434"), ("vllm_local", "http://localhost:8000")]:
        try:
            urllib.request.urlopen(host, timeout=5)
            return model_type
        except Exception:
            continue
    raise RuntimeError("No embedding server found. Start Ollama or vLLM first.")


def resolve_embedding_settings(
    model_type: str | None = None,
    *,
    config_path: Path | None = None,
) -> EmbeddingSettings:
    """
    Return API URL, key, dimension, prefix, and optional model id for the given type.

    If model_type is None, uses default_model_type from the config file.
    """
    data = _load_raw_config(config_path)
    default_type = data.get("default_model_type")
    
    key = model_type if model_type is not None else _detect_model_type()
    models = data.get("models")
    if not isinstance(models, dict):
        raise ValueError("embedding config must contain a models object")

    profile = models.get(key)
    if not isinstance(profile, dict):
        raise ValueError(f"unknown embedding model_type: {key!r}")

    try:
        url = profile["embedding_url"]
        api_key = profile["embedding_api_key"]
        dim = int(profile["embedding_dim"])
        prefix = profile["prefix"] if profile["prefix"] is not None else ""
    except KeyError as e:
        raise ValueError(f"embedding profile {key!r} missing key: {e.args[0]}") from e

    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"embedding_url for {key!r} must be a non-empty string")
    if not isinstance(api_key, str):
        raise ValueError(f"embedding_api_key for {key!r} must be a string")
    if not isinstance(prefix, str):
        raise ValueError(f"prefix for {key!r} must be a string")
    if dim < 1:
        raise ValueError(f"embedding_dim for {key!r} must be >= 1")

    emb_model = profile.get("embedding_model")
    if emb_model is not None and not isinstance(emb_model, str):
        raise ValueError(f"embedding_model for {key!r} must be a string or null")
    if isinstance(emb_model, str) and not emb_model.strip():
        emb_model = None

    return EmbeddingSettings(
        base_url=url.strip(),
        api_key=api_key,
        embedding_dim=dim,
        prefix=prefix,
        embedding_model=emb_model,
    )
