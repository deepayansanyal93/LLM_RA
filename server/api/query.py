"""
User query endpoint: retrieve context from the shared index and generate an answer.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from server.api.deps import GeneratorDep, RetrieverDep
from server.ingestion.pipeline import process_query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    """JSON body for ``POST /query``."""

    query: str = Field(..., description="User question", min_length=1, max_length=16_000)
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve")

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("query must not be empty or whitespace-only")
        return s


class QueryResponse(BaseModel):
    """Generated answer only (retrieval hits are not exposed)."""

    answer: str = Field(description="Model answer grounded on retrieved context")


class ErrorDetail(BaseModel):
    code: str
    message: str


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question",
    description="Runs retrieval on the indexed corpus then generation; returns the answer string only.",
    responses={
        400: {"description": "Invalid or empty query"},
        502: {"description": "Retrieval or generation failed"},
    },
)
async def post_query(
    body: QueryRequest,
    retriever: RetrieverDep,
    generator: GeneratorDep,
) -> QueryResponse:
    try:
        result = process_query(
            body.query,
            retriever,
            generator,
            top_k=body.top_k,
        )
    except Exception:
        logger.exception("Query pipeline failed for query length=%s", len(body.query))
        raise HTTPException(
            status_code=502,
            detail=ErrorDetail(
                code="query_failed",
                message="Retrieval or generation failed; see server logs for details.",
            ).model_dump(),
        ) from None

    return QueryResponse(answer=result.answer)
