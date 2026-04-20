"""
Ingestion orchestration for a single PDF file.

Flow (in order):
  1. Validate path and PDF header (validation.py).
  2. Extract layout text blocks with PyMuPDF (text_extractor.BasicTextExtractor).
  3. Turn blocks into strings suitable for the embedding model (chunks.basic_chunker).
  4. Request embeddings from the configured backend (extract_embeddings).

This module is the single entry point for “one file in → embeddings out” batch logic.
Callers include the CLI (scripts/validate_pdf.py) and future services.
"""

from __future__ import annotations

import logging
from pathlib import Path

from server.ingestion.chunks import basic_chunker
from server.ingestion.text_extractor import BasicTextExtractor
from server.ingestion.validation import PDFValidationError, validate_pdf_file
from server.models import Embedder
from server.models import Retriever
from server.models import Generator
from server.vector_store import VectorStore

# Logger name becomes "server.ingestion.pipeline" — under the "server" tree so it
# inherits DEBUG from logging_config (root alone would be WARNING-only for libraries).
logger = logging.getLogger(__name__)


def process_file(file_path: str) -> None:
    """
    Run the full ingestion path for one file: validate, extract blocks, chunk, embed.

    On validation failure: logs and returns (no exception) so one bad path does not
    abort a batch caller that loops over many files. Embedding errors are logged then
    re-raised so the caller can still treat API failures as fatal if desired.
    """
    # validate_pdf_file raises PDFValidationError for bad paths, wrong extension, or
    # missing %PDF header. We catch here so ingestion can stop cleanly for this file.
    try:
        validate_pdf_file(file_path)
    except PDFValidationError as e:
        logger.error("Validation failed for %s: %s", file_path, e)
        return

    logger.info("Validation passed: %s", file_path)

    # Stage 1: PyMuPDF block dicts (text, page_number, bbox).
    blocks = BasicTextExtractor().extract(file_path)
    metadata = [{"page_number": block["page_number"]} for block in blocks]

    # Stage 2: strings passed to the embedding API (chunking strategy lives in chunks.py).
    queries = basic_chunker(blocks)

    # Network or API errors from the embedding backend: full traceback in the log file
    # (logger.exception), then re-raise so scripts can exit non-zero or HTTP layer can 5xx.
    try:
        embedder = Embedder()
        embeddings = embedder.embed(queries)
    except Exception:
        logger.exception("Embedding extraction failed for %s", file_path)
        raise

    logger.info("Embeddings shape %s for %s", embeddings.shape, file_path)
    
    vector_store = VectorStore(
            index_path=Path("data/test_index"), 
            doc_path=Path("data/test_docs"),
            dimension=embedder.dim
        )
    vector_store.add(documents=queries, embeddings=embeddings, metadata=metadata)
    vector_store.save()
    print(f"Stored {len(queries)} documents and embeddings in the vector store.")

    # Retrieve the top 3 most relevant documents for a test query
    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    test_query = "What is the main topic of the document?"
    results = retriever.retrieve(test_query, top_k=3)
    print(f"Top 3 results for query: '{test_query}'")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Text: {result['text']}")
        print(f"Metadata: {result['metadata']}")
        print()

    # Instantiate the generator and generate a response based on the retrieved documents
    generator = Generator()
    response = generator.generate(test_query, results)
    print(f"Generated response: {response}")
