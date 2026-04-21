"""
Terminal REPL: dispatch ``/upload`` and ``/query`` to the API, print responses.

Run from repo root:

    python -m client.main
"""

from __future__ import annotations

import sys
from pathlib import Path

from client.api import ApiClient
from client.config import load_config
from client import ui


def _cmd_upload(api: ApiClient, rest: str) -> None:
    path = Path(rest.strip()).expanduser()
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        return
    if path.suffix.lower() != ".pdf":
        print("Only .pdf uploads are supported.", file=sys.stderr)
        return
    ui.print_user_line(f"/upload {path}")
    status, data = api.upload_pdf(path)
    if status == 200:
        ui.print_server_upload(data)
    else:
        ui.print_http_error(status, data)


def _cmd_query(api: ApiClient, rest: str) -> None:
    text = rest.strip()
    if not text:
        print("Usage: /query <your question>", file=sys.stderr)
        return
    ui.print_user_line(f"/query {text}")
    status, data = api.post_query(text)
    if status == 200:
        ui.print_server_query(data)
    else:
        ui.print_http_error(status, data)


def main() -> None:
    config = load_config()
    api = ApiClient(config)
    ui.print_banner(config)
    try:
        while True:
            try:
                line = input("> ").rstrip("\n")
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            raw = line.strip()
            if not raw:
                continue
            lower = raw.lower()
            if lower in ("/quit", "/exit", "/q"):
                print("Bye.")
                break
            if lower == "/help" or lower == "/?":
                ui.print_banner(config)
                continue
            if raw.lower().startswith("/upload"):
                rest = raw[len("/upload") :].strip()
                if not rest:
                    print("Usage: /upload </path/to/file.pdf>", file=sys.stderr)
                    continue
                _cmd_upload(api, rest)
                continue
            if raw.lower().startswith("/query"):
                rest = raw[len("/query") :].strip()
                _cmd_query(api, rest)
                continue
            print("Unknown input. Use /upload, /query, /help, or /quit.", file=sys.stderr)
    finally:
        api.close()


if __name__ == "__main__":
    main()
