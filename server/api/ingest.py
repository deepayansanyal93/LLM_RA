"""
PDF upload ingestion endpoint.

POST /upload — multipart field ``file`` (single PDF). Maximum body size per upload: 5 MiB.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])

# Hard limit for a single uploaded PDF (bytes).
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def _project_root() -> Path:
    """Repo root (parent of ``server/``)."""
    return Path(__file__).resolve().parent.parent.parent


def _tmp_files_dir() -> Path:
    return _project_root() / "tmp" / "files"


def _stored_filename(client_name: str) -> str:
    """Unique name under tmp/files; avoids path injection from client-supplied names."""
    base = Path(client_name).name.strip() or "upload.pdf"
    stem = Path(base).stem[:100] or "file"
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return f"{uuid.uuid4().hex}_{stem}.pdf"


class UploadSuccessResponse(BaseModel):
    """Returned on 200 when the upload is accepted and processed."""

    filename: str = Field(description="Original filename from the client")
    size_bytes: int = Field(description="Size of the accepted file in bytes")
    chunks: int | None = Field(
        default=None,
        description="Number of text segments sent to embedding (set when pipeline runs)",
    )
    embedding_dim: int | None = Field(
        default=None,
        description="Embedding vector dimension (set when pipeline runs)",
    )


class ErrorDetail(BaseModel):
    """Structured error payload in JSON ``detail`` for some 4xx responses."""

    code: str
    message: str


async def _read_body_with_limit(upload: UploadFile, max_bytes: int) -> bytes:
    """Read upload in chunks; raise HTTP 413 if total size exceeds ``max_bytes``."""
    total = 0
    chunks: list[bytes] = []
    chunk_size = min(1024 * 1024, max_bytes + 1)

    while True:
        piece = await upload.read(chunk_size)
        if not piece:
            break
        total += len(piece)
        if total > max_bytes:
            logger.warning(
                "Upload rejected: size over limit (%s > %s bytes)",
                total,
                max_bytes,
            )
            raise HTTPException(
                status_code=413,
                detail=ErrorDetail(
                    code="payload_too_large",
                    message=f"File exceeds maximum size of {max_bytes} bytes ({max_bytes // (1024 * 1024)} MiB)",
                ).model_dump(),
            )
        chunks.append(piece)

    return b"".join(chunks)


@router.post(
    "/upload",
    response_model=UploadSuccessResponse,
    summary="Upload a PDF",
    description=(
        "Accepts one PDF as multipart form field `file`. "
        f"Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB per request."
    ),
    responses={
        413: {"description": "File larger than 5 MiB"},
        422: {"description": "Missing or invalid multipart payload"},
    },
)
async def upload_pdf(file: Annotated[UploadFile, File(description="PDF file to ingest")]) -> UploadSuccessResponse:
    """
    Accept a single PDF upload, enforce size limit, return contract fields.

    Pipeline execution (chunks / embeddings) will populate ``chunks`` and
    ``embedding_dim`` in a follow-up change.
    """
    raw = await _read_body_with_limit(file, MAX_UPLOAD_BYTES)
    name = file.filename or "upload.pdf"

    if not raw:
        raise HTTPException(
            status_code=400,
            detail=ErrorDetail(
                code="empty_file",
                message="Uploaded file is empty",
            ).model_dump(),
        )

    tmp_dir = _tmp_files_dir()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    stored_name = _stored_filename(name)
    dest = tmp_dir / stored_name
    try:
        dest.write_bytes(raw)
    except OSError:
        logger.exception("Failed to write upload to %s", dest)
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                code="storage_failed",
                message="Could not save uploaded file",
            ).model_dump(),
        ) from None

    logger.info("Accepted upload name=%s size=%s stored=%s", name, len(raw), dest)

    return UploadSuccessResponse(
        filename=name,
        size_bytes=len(raw),
        chunks=None,
        embedding_dim=None,
    )
