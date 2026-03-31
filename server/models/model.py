from __future__ import annotations

from typing import List

import numpy as np
from openai import OpenAI

from server.config import resolve_embedding_settings


def extract_embeddings(
    queries: List[str],
    model_type: str | None = None,
    batch_size: int | None = None,
) -> np.ndarray:
    """
    Embed strings using OpenAI-compatible API settings for the given model_type.

    When model_type is None, uses default_model_type from server/config/embedding.json.
    batch_size: if a valid int in [min_batch_size, max_batch_size], use it as the
    number of texts per API request; otherwise use settings.default_batch_size.
    """
    settings = resolve_embedding_settings(model_type)
    batch_size = batch_size if \
        isinstance(batch_size, int) and not isinstance(batch_size, bool) and \
        settings.min_batch_size <= batch_size <= settings.max_batch_size \
            else settings.default_batch_size
    
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

    if n == 0:
        return ret

    num_chunks = (n + batch_size - 1) // batch_size

    for chunk_index in range(num_chunks):
        start = chunk_index * batch_size
        end = min(start + batch_size, n)
        batch_inputs = prefixed[start:end]
        responses = client.embeddings.create(input=batch_inputs, model=model)
        m = end - start

        if len(responses.data) != m:
            raise RuntimeError(
                f"expected {m} embedding rows for batch [{start}:{end}), "
                f"got {len(responses.data)}"
            )

        ordered = sorted(responses.data, key=lambda d: d.index)
        for j in range(m):
            vec = np.array(ordered[j].embedding, dtype=np.float64)
            if vec.shape[0] != dim:
                raise RuntimeError(
                    f"expected embedding dim {dim}, got {vec.shape[0]} for row {start + j}"
                )
            ret[start + j] = vec

    return ret


if __name__ == "__main__":
    extract_embeddings(["Follow the white rabbit.", "Follow the black rabbit."])
