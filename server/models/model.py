from __future__ import annotations

from typing import List

import numpy as np
from openai import OpenAI

from server.config import resolve_embedding_settings


def extract_embeddings(
    queries: List[str],
    model_type: str | None = None,
) -> np.ndarray:
    """
    Embed strings using OpenAI-compatible API settings for the given model_type.

    When model_type is None, uses default_model_type from server/config/embedding.json.
    """
    settings = resolve_embedding_settings(model_type)

    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
    )

    if settings.embedding_model is not None:
        model = settings.embedding_model
    else:
        listed = client.models.list()
        if not listed.data:
            raise RuntimeError("embeddings API returned no models; set embedding_model in config")
        model = listed.data[0].id

    n = len(queries)
    dim = settings.embedding_dim
    ret = np.zeros((n, dim))

    prefixed = [f"{settings.prefix}{q}" for q in queries] if settings.prefix else list(queries)
    responses = client.embeddings.create(input=prefixed, model=model)

    if len(responses.data) != n:
        raise RuntimeError(
            f"expected {n} embedding rows, got {len(responses.data)}"
        )

    for i in range(n):
        vec = np.array(responses.data[i].embedding, dtype=np.float64)
        if vec.shape[0] != dim:
            raise RuntimeError(
                f"expected embedding dim {dim}, got {vec.shape[0]} for row {i}"
            )
        ret[i] = vec

    return ret


if __name__ == "__main__":
    extract_embeddings(["Follow the white rabbit.", "Follow the black rabbit."])
