"""HTTP calls to ``POST /upload`` and ``POST /query``."""

from __future__ import annotations

from pathlib import Path

import httpx

from client.config import ClientConfig


class ApiClient:
    """Thin wrapper around httpx for the LLM_RA FastAPI server."""

    def __init__(self, config: ClientConfig) -> None:
        self._config = config
        self._client = httpx.Client(base_url=config.base_url, timeout=config.timeout_s)

    def close(self) -> None:
        self._client.close()

    def upload_pdf(self, pdf_path: Path) -> tuple[int, dict]:
        """
        POST multipart ``file`` to ``/upload``.

        Returns ``(status_code, body)`` where ``body`` is parsed JSON if possible,
        else ``{"raw": text}``.
        """
        name = pdf_path.name
        with pdf_path.open("rb") as f:
            files = {"file": (name, f, "application/pdf")}
            r = self._client.post("/upload", files=files)
        return _response_payload(r)

    def post_query(self, query: str, top_k: int = 5) -> tuple[int, dict]:
        """POST JSON to ``/query``."""
        r = self._client.post("/query", json={"query": query, "top_k": top_k})
        return _response_payload(r)


def _response_payload(r: httpx.Response) -> tuple[int, dict]:
    try:
        body = r.json()
        if isinstance(body, dict):
            return r.status_code, body
    except Exception:
        pass
    return r.status_code, {"raw": r.text}
