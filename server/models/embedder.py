from __future__ import annotations

from typing import List

import numpy as np
from openai import OpenAI

from server.config import resolve_embedding_settings

"""Module for embedding text."""

class Embedder:
    def __init__(self, model_type: str | None = None, batch_size: int | None = None):
        self.model_type = model_type
        self.settings = resolve_embedding_settings(self.model_type)
        self.batch_size = batch_size if \
            isinstance(batch_size, int) and not isinstance(batch_size, bool) and \
                self.settings.min_batch_size <= batch_size <= self.settings.max_batch_size \
                    else self.settings.default_batch_size

        self.client = OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
        )
        if self.settings.embedding_model is not None:
            self.model = self.settings.embedding_model
        else:
            # TODO: Isn't there a better way to get a default embedding model?
            listed = self.client.models.list()
            if not listed.data:
                raise RuntimeError("embeddings API returned no models; set embedding_model in config")
            self.model = listed.data[0].id

        self.dim = self.settings.embedding_dim

    def embed(self, queries: List[str], batch_size: int | None = None) -> np.ndarray:

        n = len(queries)
        
        dim = self.settings.embedding_dim
        ret = np.zeros((n, dim))
        if n == 0:
            return ret

        # Get embeddings from the API. If settings.prefix is set, prepend it to each query before sending to the API.
        prefixed = [f"{self.settings.prefix}{q}" for q in queries] if self.settings.prefix else list(queries)
        batch_size = batch_size if batch_size and \
            isinstance(batch_size, int) and not isinstance(batch_size, bool) and \
                self.settings.min_batch_size <= batch_size <= self.settings.max_batch_size \
                    else self.batch_size
        
        num_chunks = (n + batch_size - 1) // batch_size
        for chunk_index in range(num_chunks):
            start = chunk_index * batch_size
            end = min(start + batch_size, n)
            batch_inputs = prefixed[start:end]
            responses = self.client.embeddings.create(input=batch_inputs, model=self.model)
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
    embedder = Embedder()
    embeddings = embedder.embed(["Follow the white rabbit.", "Follow the black rabbit."])
