"""
ASGI application for the API server.

Run with: uvicorn server.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from server.api.ingest import router as ingest_router
from server.logging_config import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    yield


app = FastAPI(
    title="LLM RA API",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(ingest_router)
