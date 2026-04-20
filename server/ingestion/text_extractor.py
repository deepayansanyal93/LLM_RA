"""
Stage 1 — PDF layout text blocks (PyMuPDF).

Opens the PDF and collects native text blocks from each page: stripped text,
page index, and bounding box. This is layout-level extraction, not the final
shape for LLM / embedding input.

Stage 2 (strings for embeddings) lives in chunks.py — e.g. basic_chunker maps
blocks to query strings.

Callers must validate the file (e.g. validate_pdf_file in pipeline) before
extract(); this module does not perform path or PDF header checks.
"""

from pathlib import Path
from typing import Any

import pymupdf


class BasicTextExtractor:
    """Extract PyMuPDF text blocks from a PDF (stage 1 of text ingestion)."""

    def extract(self, file_path: str | Path) -> list[dict[str, Any]]:
        """
        Return all text blocks from the PDF.

        Does not validate the path or file; the pipeline does that first.

        Each block is a dict with:
        - "text": stripped text content
        - "page_number": 1-based page index
        - "bbox": (x0, y0, x1, y1) in PDF coordinates

        Raises:
            pymupdf.PdfError: If the PDF cannot be opened or is corrupted.
        """
        doc = pymupdf.open(file_path)
        blocks: list[dict[str, Any]] = []

        try:
            for page_index, page in enumerate(doc):
                page_blocks = page.get_text("blocks")
                for block in page_blocks:
                    x0, y0, x1, y1, text, block_no, block_type = block
                    text = text.strip() if text else ""
                    if block_type != 0 or len(text) == 0:
                        continue
                    blocks.append(
                        {
                            "text": text,
                            "page_number": page_index + 1,
                            "bbox": (x0, y0, x1, y1),
                        }
                    )
        finally:
            doc.close()

        return blocks
