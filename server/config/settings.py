"""Load model settings from the packaged JSON config."""

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
    default_batch_size: int
    min_batch_size: int
    max_batch_size: int


@dataclass(frozen=True)
class GeneratorSettings:
    """Resolved generator client parameters for one model type."""

    base_url: str
    api_key: str
    generation_model: str | None


def _load_raw_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or _CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def _detect_model_type(models) -> str:
    import urllib.request
    possible_models = []
    for model_name in models:
        model_url = models[model_name].get("check_url")
        possible_models.append((model_name, model_url))
        
    for model_name, model_url in possible_models:
        try:
            print("model_url:", model_url)
            urllib.request.urlopen(model_url, timeout=5)
            return model_name
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

    Args:
        model_type: Explicit model type key, or None to auto-detect.
        config_path: Optional path to the config file.

    Returns:
        EmbeddingSettings for the resolved model type.

    Raises:
        ValueError: If the config is invalid or required fields are missing.
    """
    data = _load_raw_config(config_path)
    min_bs = data.get("min_batch_size")
    if not isinstance(min_bs, int) or isinstance(min_bs, bool):
        raise ValueError("min_batch_size must be an int between 1 and 512")

    max_bs = data.get("max_batch_size")
    if not isinstance(max_bs, int) or isinstance(max_bs, bool):
        raise ValueError("max_batch_size must be an int between 1 and 512")

    if min_bs > max_bs:
        raise ValueError("min_batch_size must be <= max_batch_size")

    bs = data.get("default_batch_size")
    if not isinstance(bs, int):
        raise ValueError("default_batch_size must be an int")
    if not (min_bs <= bs <= max_bs):
        raise ValueError(
            "default_batch_size must be between min_batch_size and max_batch_size"
        )

    
    models = data.get("models")
    if not isinstance(models, dict):
        raise ValueError("embedding config must contain a models object")
    
    key = model_type if model_type is not None else _detect_model_type(models)

    profile = models.get(key)
    if not isinstance(profile, dict):
        raise ValueError(f"unknown embedding model_type: {key!r}")

    try:
        dim = int(profile["embedding_dim"])
        prefix = profile["prefix"] if profile["prefix"] is not None else ""
        url = profile["embedding_url"]
        api_key = profile["embedding_api_key"]
    except KeyError as e:
        raise ValueError(f"embedding profile {key!r} missing key: {e.args[0]}") from e

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
        base_url=url,
        api_key=api_key,
        embedding_dim=dim,
        prefix=prefix,
        embedding_model=emb_model,
        default_batch_size=bs,
        min_batch_size=min_bs,
        max_batch_size=max_bs,
    )


def resolve_generator_settings(
    model_type: str | None = None,
    *,
    config_path: Path | None = None,
) -> GeneratorSettings:
    """
    Return API URL, key, and optional generation model id for the given type.

    Args:
        model_type: Explicit model type key, or None to auto-detect.
        config_path: Optional path to the config file.

    Returns:
        GeneratorSettings for the resolved model type.

    Raises:
        ValueError: If the config is invalid or required fields are missing.
    """
    key, url, api_key, profile = _load_profile(model_type, config_path)
    gen_model = _normalize_model_str(profile.get("generation_model"), "generation_model", key)

    return GeneratorSettings(
        base_url=url,
        api_key=api_key,
        generation_model=gen_model,
    )
