"""
Ingestion orchestration for a single PDF file.

Flow (in order):
  1. Validate path and PDF header (validation.py).
  2. Extract layout text blocks with PyMuPDF (text_extractor.BasicTextExtractor).
  3. Turn blocks into strings suitable for the embedding model (chunks.basic_chunker).
  4. Request embeddings from the configured backend (embeddings.extract_embeddings).

This module is the single entry point for “one file in → embeddings out” batch logic.
Callers include the CLI (scripts/validate_pdf.py) and future services.
"""

import sys

from server.embeddings.embeddings import extract_embeddings
from server.ingestion.chunks import basic_chunker
from server.ingestion.text_extractor import BasicTextExtractor
from server.ingestion.validation import PDFValidationError, validate_pdf_file


def process_file(file_path: str) -> None:
    """
    Run the full ingestion path for one file: validate, extract blocks, chunk, embed.

    On validation failure, prints to stderr and returns without extracting.
    """
    try:
        validate_pdf_file(file_path)
    except PDFValidationError as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        return

    print(f"Validation passed: {file_path}")

    # Stage 1: PyMuPDF block dicts (text, page_number, bbox).
    blocks = BasicTextExtractor().extract(file_path)

    # Stage 2: strings passed to the embedding API (chunking strategy lives in chunks.py).
    queries = basic_chunker(blocks)

    embeddings = extract_embeddings(queries)
    print(embeddings.shape)
