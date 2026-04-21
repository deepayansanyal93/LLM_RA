"""
CLI entry point for the PDF ingestion pipeline.

Delegates to server.ingestion.pipeline.process_file, then runs a fixed demo
query via process_query (same pattern as scripts/run_pipeline.py).

Usage:
    python scripts/validate_pdf.py <file_path>

Example:
    python scripts/validate_pdf.py /path/to/document.pdf

Run from the project root directory.
"""

import sys
from pathlib import Path
import shutil

# Add the project root to the path so we can import from server
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from server.config.vector_store_paths import load_vector_store_paths
from server.logging_config import configure_logging
from server.ingestion.pipeline import process_file, process_query
from server.models import Embedder, Generator, Retriever
from server.vector_store import VectorStore

configure_logging()


def main() -> None:
    """Read file path from argv and run the full ingestion pipeline for that file."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_pdf.py <file_path>", file=sys.stderr)
        sys.exit(1)

    embedder = Embedder()
    index_path, doc_path = load_vector_store_paths(project_root)
    shutil.rmtree(index_path, ignore_errors=True)
    shutil.rmtree(doc_path, ignore_errors=True)

    vector_store = VectorStore(
        index_path=index_path,
        doc_path=doc_path,
        dimension=embedder.dim,
    )

    print(sys.argv[1])
    result = process_file(sys.argv[1], embedder, vector_store)
    if not result.ok:
        sys.exit(1)
    print(f"Ingested {result.chunks_stored} chunks.")

    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    generator = Generator()
    test_query = "What is the main topic of the document?"
    qr = process_query(test_query, retriever, generator, top_k=3)
    print(f"Top retrieval for query: '{test_query}'")
    for i, hit in enumerate(qr.retrieval_results):
        print(f"Result {i + 1}:")
        print(f"Text: {hit['text']}")
        print(f"Metadata: {hit['metadata']}")
        print()
    print(f"Generated response: {qr.answer}")


if __name__ == "__main__":
    main()
