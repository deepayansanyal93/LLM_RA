"""
CLI script to validate a PDF and run ingestion (and an optional demo query).

Usage:
    python scripts/run_pipeline.py <file_path> [--reset]

Example:
    python scripts/run_pipeline.py /path/to/document.pdf

Run from the project root directory.
"""

import argparse
import shutil
import sys
from pathlib import Path

# Add the project root to the path so we can import from server
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from server.config.vector_store_paths import load_vector_store_paths
from server.ingestion.pipeline import process_file, process_query
from server.models import Embedder, Generator, Retriever
from server.vector_store import VectorStore


def main() -> None:
    """Read file path from command line, run ingest, then a demo query."""
    parser = argparse.ArgumentParser(
        description="Validate a PDF and run the ingestion pipeline.",
        usage="python scripts/run_pipeline.py <file_path> [--reset]",
    )
    parser.add_argument("file_path", help="Path to the PDF file to process.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing index and docstore directories before ingesting.",
    )
    args = parser.parse_args()

    file_path = args.file_path
    embedder = Embedder()
    index_path, doc_path = load_vector_store_paths(project_root)

    if args.reset:
        shutil.rmtree(index_path, ignore_errors=True)
        shutil.rmtree(doc_path, ignore_errors=True)

    try:
        vector_store = VectorStore(
            index_path=index_path,
            doc_path=doc_path,
            dimension=embedder.dim,
        )
        result = process_file(file_path, embedder, vector_store)
        if not result.ok:
            print("Validation failed or file could not be ingested.", file=sys.stderr)
            sys.exit(1)
        print(f"Stored {result.chunks_stored} chunks for {file_path}.")

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

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
