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

from server.embeddings.embeddings import extract_embeddings
from server.ingestion.chunks import basic_chunker
from server.ingestion.text_extractor import BasicTextExtractor
from server.ingestion.validation import PDFValidationError, validate_pdf_file

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

    # Stage 2: strings passed to the embedding API (chunking strategy lives in chunks.py).
    queries = basic_chunker(blocks)

    # Network or API errors from the embedding backend: full traceback in the log file
    # (logger.exception), then re-raise so scripts can exit non-zero or HTTP layer can 5xx.
    try:
        embeddings = extract_embeddings(queries)
    except Exception:
        logger.exception("Embedding extraction failed for %s", file_path)
        raise

    logger.info("Embeddings shape %s for %s", embeddings.shape, file_path)
