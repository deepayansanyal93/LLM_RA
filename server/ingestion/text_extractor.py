"""
Text extraction from PDF files.

Extracts all text blocks from a PDF using PyMuPDF. Each block contains
text content, page number, and bounding box coordinates for downstream
use (e.g. proximity-based linking, layout analysis).
"""

from pathlib import Path
from typing import Any

import pymupdf

from server.ingestion.validation import validate_pdf_file


def extract_text_blocks(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Extract all text blocks from a PDF file.

    Each block is returned as a dict with:
    - "text": extracted text content
    - "page_number": 1-based page index
    - "bbox": bounding box (x0, y0, x1, y1) for layout analysis

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of text block dicts, one per block across all pages.

    Raises:
        PDFValidationError: If the file fails validation.
        pymupdf.PdfError: If the PDF cannot be opened or is corrupted.
    """
    # Step 1: Validate the input file before opening
    validate_pdf_file(file_path)

    # Step 2: Open the PDF document (fitz is the PyMuPDF module name)
    doc = pymupdf.open(file_path)

    blocks: list[dict[str, Any]] = []

    try:
        # Step 3: Iterate over all pages
        for page_index, page in enumerate(doc):
            # Step 4: Get text blocks for this page
            # PyMuPDF returns: (x0, y0, x1, y1, "text", block_no, block_type)
            # block_type: 0=text, 1=image, 3=vector
            page_blocks = page.get_text("blocks")

            # Step 5: Convert each block to a dict with text, page_number, and bbox
            for block in page_blocks:
                x0, y0, x1, y1, text, block_no, block_type = block
                text = text.strip() if text else ""
                # Include only text blocks (type 0); skip image and vector blocks
                if block_type != 0 or len(text) == 0:
                    continue
                
                block_dict = {
                    "text": text.strip() if text else "",
                    "page_number": page_index + 1,
                    # "bbox": (x0, y0, x1, y1),
                }
                blocks.append(block_dict)
    finally:
        # Step 6: Always close the document to free resources
        doc.close()

    return blocks

