"""
FastAPI dependencies that read shared services from ``app.state``.

Populated in ``server.main`` lifespan: ``embedder``, ``vector_store``,
``vector_store_lock``, ``project_root``, ``generator``, ``retriever``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from server.models import Embedder, Generator, Retriever
from server.vector_store import VectorStore


def _require_state_attr(request: Request, name: str):
    value = getattr(request.app.state, name, None)
    if value is None:
        raise HTTPException(
            status_code=503,
            detail=f"Application state '{name}' is not initialized",
        )
    return value


def get_embedder(request: Request) -> Embedder:
    return _require_state_attr(request, "embedder")


def get_vector_store(request: Request) -> VectorStore:
    return _require_state_attr(request, "vector_store")


def get_retriever(request: Request) -> Retriever:
    return _require_state_attr(request, "retriever")


def get_generator(request: Request) -> Generator:
    return _require_state_attr(request, "generator")


def get_vector_store_lock(request: Request) -> asyncio.Lock:
    return _require_state_attr(request, "vector_store_lock")


def get_project_root(request: Request) -> Path:
    return _require_state_attr(request, "project_root")


EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
RetrieverDep = Annotated[Retriever, Depends(get_retriever)]
GeneratorDep = Annotated[Generator, Depends(get_generator)]
VectorStoreLockDep = Annotated[asyncio.Lock, Depends(get_vector_store_lock)]
ProjectRootDep = Annotated[Path, Depends(get_project_root)]

__all__ = [
    "EmbedderDep",
    "GeneratorDep",
    "ProjectRootDep",
    "RetrieverDep",
    "VectorStoreDep",
    "VectorStoreLockDep",
    "get_embedder",
    "get_generator",
    "get_project_root",
    "get_retriever",
    "get_vector_store",
    "get_vector_store_lock",
]
