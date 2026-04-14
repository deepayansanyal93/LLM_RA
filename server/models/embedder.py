from __future__ import annotations

from typing import List

import numpy as np
from openai import OpenAI

from server.config import resolve_embedding_settings

"""Module for embedding text."""

class Embedder:
    def __init__(self, model_type: str | None = None):
        self.model_type = model_type
        self.settings = resolve_embedding_settings(self.model_type)

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

    def embed(self, queries: List[str]) -> np.ndarray:

        n = len(queries)
        ret = []

        # Get embeddings from the API. If settings.prefix is set, prepend it to each query before sending to the API.
        prefixed = [f"{self.settings.prefix}{q}" for q in queries] if self.settings.prefix else list(queries)
        responses = self.client.embeddings.create(input=prefixed, model=self.model)

        if len(responses.data) != n:
            raise RuntimeError(
                f"expected {n} embedding rows, got {len(responses.data)}"
            )

        for i in range(n):
            vec = np.array(responses.data[i].embedding, dtype=np.float64)
            if vec.shape[0] != self.dim:
                raise RuntimeError(
                    f"expected embedding dim {self.dim}, got {vec.shape[0]} for row {i}"
                )
            ret.append(vec)

        return np.array(ret).reshape(n, self.dim)


if __name__ == "__main__":
    embedder = Embedder()
    embeddings = embedder.embed(["Follow the white rabbit.", "Follow the black rabbit."])
