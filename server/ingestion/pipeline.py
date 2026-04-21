"""
Ingestion and query orchestration.

``process_file`` runs the PDF path through validation, layout extraction, chunking,
embedding, and persistence on a caller-supplied vector store.

``process_query`` runs RAG-style answering using a **pre-constructed** ``Retriever``
(so the same instance can be reused across requests) and a ``Generator``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from server.ingestion.chunks import basic_chunker
from server.ingestion.text_extractor import BasicTextExtractor
from server.ingestion.validation import PDFValidationError, validate_pdf_file
from server.models import Embedder, Generator, Retriever
from server.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    """Outcome of ``process_file`` for one path."""

    ok: bool
    """False when validation failed before any embedding work."""
    chunks_stored: int
    """Number of chunk strings written to the vector store (0 if not ok)."""


@dataclass(frozen=True)
class QueryResult:
    """Outcome of ``process_query`` for one user question."""

    answer: str
    retrieval_results: list[dict]


def process_file(
    file_path: str | Path,
    embedder: Embedder,
    vector_store: VectorStore,
) -> IngestResult:
    """
    Ingest one PDF: validate, extract blocks, chunk, embed, add to ``vector_store``, save.

    On validation failure: logs, returns ``IngestResult(ok=False, chunks_stored=0)``.
    Embedding failures are logged and re-raised.
    """
    path_str = str(file_path)
    try:
        validate_pdf_file(path_str)
    except PDFValidationError as e:
        logger.error("Validation failed for %s: %s", path_str, e)
        return IngestResult(ok=False, chunks_stored=0)

    logger.info("Validation passed: %s", path_str)

    blocks = BasicTextExtractor().extract(path_str)
    metadata = [{"page_number": block["page_number"]} for block in blocks]
    queries = basic_chunker(blocks)

    try:
        embeddings = embedder.embed(queries)
    except Exception:
        logger.exception("Embedding extraction failed for %s", path_str)
        raise

    logger.info("Embeddings shape %s for %s", embeddings.shape, path_str)

    vector_store.add(documents=queries, embeddings=embeddings, metadata=metadata)
    vector_store.save()
    logger.info("Stored %s chunks in vector store for %s", len(queries), path_str)

    return IngestResult(ok=True, chunks_stored=len(queries))


def process_query(
    query: str,
    retriever: Retriever,
    generator: Generator,
    *,
    top_k: int = 5,
) -> QueryResult:
    """
    Retrieve context with ``retriever`` (reuse the same instance across calls),
    then generate an answer with ``generator``.
    """
    results = retriever.retrieve(query, top_k=top_k)
    answer = generator.generate(query, results)
    return QueryResult(answer=answer, retrieval_results=results)
