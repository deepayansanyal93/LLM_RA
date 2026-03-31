"""
Stage 2 — from PyMuPDF blocks to embedding inputs.

Takes the block dicts produced by BasicTextExtractor (text, page_number, bbox)
and builds the list of strings sent to extract_embeddings / the LLM backend.

basic_chunker is the default: one embedding input per non-empty block text.
Replace or add alternatives here for merged blocks, token limits, overlap, etc.
"""

from typing import Any


def basic_chunker(blocks: list[dict[str, Any]]) -> list[str]:
    """
    Map each block's "text" field to a string in order.

    Input blocks typically come from BasicTextExtractor.extract().
    """
    queries: list[str] = []
    for block in blocks:
        queries.append(block["text"])
    return queries
