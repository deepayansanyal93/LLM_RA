"""
ASGI application for the API server.

Run with: uvicorn server.main:app --reload
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from server.api.ingest import router as ingest_router
from server.api.query import router as query_router
from server.config.vector_store_paths import load_vector_store_paths
from server.logging_config import configure_logging
from server.models import Embedder, Generator, Retriever
from server.vector_store import VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    project_root = Path(__file__).resolve().parent.parent
    index_path, doc_path = load_vector_store_paths(project_root)
    embedder = Embedder()

    def refresh_retriever() -> None:
        """Rebuild retriever so it always targets the current in-memory store after adds."""
        app.state.retriever = Retriever(
            vector_store=app.state.vector_store,
            embedder=app.state.embedder,
        )

    vector_store = VectorStore(
        index_path=index_path,
        doc_path=doc_path,
        dimension=embedder.dim,
        on_after_add=refresh_retriever,
    )
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.vector_store_lock = asyncio.Lock()
    app.state.project_root = project_root
    app.state.generator = Generator()
    refresh_retriever()
    yield


app = FastAPI(
    title="LLM RA API",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(ingest_router)
app.include_router(query_router)
