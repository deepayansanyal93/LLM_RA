"""Terminal output: banner, user/server lines, and formatted API payloads."""

from __future__ import annotations

from typing import Any

from client.config import ClientConfig


def print_banner(config: ClientConfig) -> None:
    print(f"LLM_RA chat client — API: {config.base_url}")
    print("Commands: /upload <path.pdf>  |  /query <text>  |  /help  |  /quit")
    print("-" * 60)


def print_user_line(text: str) -> None:
    print(f"\nYou: {text}")


def print_server_upload(payload: dict[str, Any]) -> None:
    print("Server (upload):")
    for key in ("filename", "size_bytes", "chunks", "embedding_dim"):
        if key in payload:
            print(f"  {key}: {payload[key]}")


def print_server_query(payload: dict[str, Any]) -> None:
    print("Server (query):")
    ans = payload.get("answer")
    if ans is not None:
        print(f"  {ans}")
    else:
        print(f"  {payload}")


def print_http_error(status: int, payload: dict[str, Any]) -> None:
    print(f"Server error ({status}):")
    detail = payload.get("detail")
    if isinstance(detail, dict):
        print(f"  code: {detail.get('code', '')}")
        print(f"  message: {detail.get('message', detail)}")
    else:
        print(f"  {payload}")
