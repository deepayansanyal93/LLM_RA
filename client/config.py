"""Client configuration: edit the module-level values below, then run the client."""

from __future__ import annotations

from dataclasses import dataclass

# --- User-editable: change here, then restart ``python -m client.main``. ---
BASE_URL = "http://127.0.0.1:9000"
HTTP_TIMEOUT_S = 600.0
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClientConfig:
    """Resolved settings for ``ApiClient`` and the REPL."""

    base_url: str
    """API root (no trailing slash)."""
    timeout_s: float
    """HTTP timeout in seconds for upload and query calls."""


def load_config() -> ClientConfig:
    """Build ``ClientConfig`` from the module-level ``BASE_URL`` and ``HTTP_TIMEOUT_S``."""
    url = BASE_URL.strip().rstrip("/")
    return ClientConfig(base_url=url, timeout_s=float(HTTP_TIMEOUT_S))
