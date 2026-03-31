"""
CLI script to validate a PDF file.

Usage:
    python scripts/validate_pdf.py <file_path>

Example:
    python scripts/validate_pdf.py /path/to/document.pdf

Run from the project root directory.
"""

import sys
from pathlib import Path

# Add the project root to the path so we can import from server
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from server.ingestion.validation import PDFValidationError, validate_pdf_file
from server.ingestion.text_extractor import extract_text_blocks
from server.embeddings.embeddings import extract_embeddings


def main() -> None:
    """Read file path from command line, validate the PDF, and print the result."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_pdf.py <file_path>", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]
    print(sys.argv[1])
    try:
        validate_pdf_file(file_path)
        print(f"Validation passed: {file_path}")
        blocks = extract_text_blocks(file_path)
        queries = []
        for block in blocks:
            queries.append(block["text"])
        embeddings = extract_embeddings(queries)
        print(embeddings.shape)
    except PDFValidationError as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
