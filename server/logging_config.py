"""
Application-wide logging setup.

Root logger is WARNING (default deny for INFO/DEBUG from third-party libraries).
The ``server`` package is set to DEBUG so application loggers (server.*) can use
INFO and DEBUG; those records still pass each handler's level (file DEBUG, stderr INFO).

Uses the standard library logging module. For structured production logs, consider
structlog on top of these handlers.

Call configure_logging() once at process startup (CLI or API server).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

# Guard: logging.basicConfig or adding handlers twice would duplicate every log line.
_CONFIGURED = False
# Set on first configure_logging() call; useful for tests or diagnostics (optional).
_log_file_path: Path | None = None


def configure_logging() -> Path | None:
    """
    Attach a timestamped file under project_root/logs/ and stderr to the root logger.

    Root is WARNING so third-party libraries (httpx, httpcore, etc.) do not emit
    routine INFO/DEBUG. Only the ``server`` package tree is lowered to DEBUG so
    application INFO/DEBUG reaches the file; stderr stays at INFO via its handler.

    Safe to call multiple times; only the first call installs handlers.
    Returns the log file path on first call, None on subsequent calls.
    """
    global _CONFIGURED, _log_file_path
    if _CONFIGURED:
        return None

    # This file lives in server/; repo root is one level up (where logs/ should live).
    project_root = Path(__file__).resolve().parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # One file per process start avoids mixing runs; second-precision is enough for local dev.
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = logs_dir / f"ingest_{stamp}.log"
    _log_file_path = log_path

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    # Inherited by httpx, openai, etc. when their loggers use NOTSET: they won't emit INFO.
    root.setLevel(logging.WARNING)

    # File: capture full detail from whitelisted server.* loggers (DEBUG and up).
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console: slightly quieter so operators see INFO from the app, not DEBUG noise.
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    # Child loggers named server.* have NOTSET; they inherit this DEBUG effective level,
    # so isEnabledFor(INFO/DEBUG) is true for the app while libraries stay at WARNING.
    logging.getLogger("server").setLevel(logging.DEBUG)

    _CONFIGURED = True
    logging.getLogger(__name__).debug("Logging to %s", log_path)
    return log_path
