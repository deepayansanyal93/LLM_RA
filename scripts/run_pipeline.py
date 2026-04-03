"""
CLI script to validate a PDF file.

Usage:
    python scripts/run_pipeline.py <file_path> [--reset]

Example:
    python scripts/run_pipeline.py /path/to/document.pdf

Run from the project root directory.
"""

import sys
import argparse
import shutil
from pathlib import Path

# Add the project root to the path so we can import from server
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from server.ingestion.validation import PDFValidationError, validate_pdf_file
from server.ingestion.text_extractor import extract_text_blocks
from server.models import Embedder, Retriever, Generator
from server.vector_store import VectorStore


def main() -> None:
    """Read file path from command line, validate the PDF, and print the result."""
    parser = argparse.ArgumentParser(
        description="Validate a PDF and run the ingestion pipeline.",
        usage="python scripts/run_pipeline.py <file_path> [--reset]",
    )
    parser.add_argument("file_path", help="Path to the PDF file to process.")                                                           
    parser.add_argument("--reset", action="store_true", help="Clear existing index and docstore before ingesting.")                     
    args = parser.parse_args()                                                                                                          
                                                                                                                                      
    file_path = args.file_path                                                                                                          
    if args.reset:                                                                                                                      
        shutil.rmtree("data/test_index", ignore_errors=True)                                                                            
        shutil.rmtree("data/test_docs", ignore_errors=True) 

    try:
        # Validate the PDF file
        validate_pdf_file(file_path)
        print(f"Validation passed: {file_path}")

        # Extract text blocks and metadata from the PDF
        blocks = extract_text_blocks(file_path)
        queries = [block["text"] for block in blocks]
        metadata = [{"page_number": block["page_number"]} for block in blocks]

        # instantiate the embedder and get embeddings for the extracted text blocks
        embedder = Embedder()
        embeddings = embedder.embed(queries)
        print(embeddings.shape)

        # instantiate the vector store and add the documents, embeddings, and metadata
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
 

    except PDFValidationError as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
