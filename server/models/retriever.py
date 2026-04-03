from .embedder import Embedder
from server.vector_store import VectorStore

"""Module for retrieving relevant documents based on a query."""

class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.embedder.embed([query])[0]
        results = self.vector_store.search(query_embedding, top_k)
        return results