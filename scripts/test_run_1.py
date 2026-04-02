"""
CLI entry point for the PDF ingestion pipeline.

Delegates to server.ingestion.pipeline.process_file, which validates the file,
extracts PyMuPDF blocks, chunks text for embeddings, and calls the embedding
backend. This script only parses argv and invokes that flow.

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


from server.ingestion.pipeline import process_file

def main() -> None:
    """Read file path from argv and run the full ingestion pipeline for that file."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_pdf.py <file_path>", file=sys.stderr)
        sys.exit(1)

    print(sys.argv[1])
    process_file(sys.argv[1])


if __name__ == "__main__":
    main()
