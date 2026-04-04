"""Load model settings from the packaged JSON config."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


@dataclass(frozen=True)
class EmbeddingSettings:
    """Resolved embedding client parameters for one model type."""

    base_url: str
    api_key: str
    embedding_dim: int
    prefix: str
    embedding_model: str | None


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


def _detect_model_type() -> str:
    import urllib.request
    for model_type, host in [("ollama_local", "http://localhost:11434"), ("vllm_local", "http://localhost:8000")]:
        try:
            urllib.request.urlopen(host, timeout=5)
            return model_type
        except Exception:
            continue
    raise RuntimeError("No embedding server found. Start Ollama or vLLM first.")


def _load_profile(model_type: str | None, config_path: Path | None) -> tuple[str, str, dict]:
    """
    Resolve model type, load config, and return (key, url, api_key, profile).

    Args:
        model_type: Explicit model type key, or None to auto-detect.
        config_path: Optional path to the config file.

    Returns:
        Tuple of (resolved key, base_url, api_key, full profile dict) as a 4-tuple.

    Raises:
        ValueError: If the config structure is invalid or the model type is unknown.
    """
    data = _load_raw_config(config_path)
    key = model_type if model_type is not None else _detect_model_type()

    models = data.get("models")
    if not isinstance(models, dict):
        raise ValueError("config must contain a models object")

    profile = models.get(key)
    if not isinstance(profile, dict):
        raise ValueError(f"unknown model_type: {key!r}")

    try:
        url = profile["url"]
        api_key = profile["api_key"]
    except KeyError as e:
        raise ValueError(f"profile {key!r} missing key: {e.args[0]}") from e

    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"url for {key!r} must be a non-empty string")
    if not isinstance(api_key, str):
        raise ValueError(f"api_key for {key!r} must be a string")

    return key, url.strip(), api_key, profile


def _normalize_model_str(value: Any, field: str, key: str) -> str | None:
    """
    Validate and normalise an optional model name string from a profile.

    Args:
        value: The raw value from the profile dict.
        field: The field name, used in error messages.
        key: The model type key, used in error messages.

    Returns:
        Stripped string, or None if value is None or blank.

    Raises:
        ValueError: If value is neither a string nor None.
    """
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{field} for {key!r} must be a string or null")
    if isinstance(value, str) and not value.strip():
        return None
    return value


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
    key, url, api_key, profile = _load_profile(model_type, config_path)

    try:
        dim = int(profile["embedding_dim"])
        prefix = profile["prefix"] if profile["prefix"] is not None else ""
    except KeyError as e:
        raise ValueError(f"embedding profile {key!r} missing key: {e.args[0]}") from e

    if not isinstance(prefix, str):
        raise ValueError(f"prefix for {key!r} must be a string")
    if dim < 1:
        raise ValueError(f"embedding_dim for {key!r} must be >= 1")

    emb_model = _normalize_model_str(profile.get("embedding_model"), "embedding_model", key)

    return EmbeddingSettings(
        base_url=url,
        api_key=api_key,
        embedding_dim=dim,
        prefix=prefix,
        embedding_model=emb_model,
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
