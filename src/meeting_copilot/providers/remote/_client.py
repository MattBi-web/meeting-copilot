"""Shared HTTP client per i remote providers.

Singleton httpx.Client con:
- keep-alive (riusa connessione TCP)
- timeout config-driven
- Bearer auth header
- retry leggero su connection error / 5xx

Eventi SSE: parser custom per leggere `data: {...}\\n\\n` line-by-line.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from threading import Lock
from typing import Any

try:
    import httpx
except ImportError as e:
    raise RuntimeError(
        "httpx non installato. Installa con: uv pip install httpx"
    ) from e

from ...config import SERVER


_client: httpx.Client | None = None
_lock = Lock()


def _resolve_base_url() -> str:
    return os.environ.get("MEETING_COPILOT_SERVER_URL", SERVER.client_base_url).rstrip("/")


def _resolve_api_key() -> str:
    return os.environ.get("MEETING_COPILOT_API_KEY", SERVER.api_key)


def get_client() -> httpx.Client:
    """Ritorna il singleton httpx.Client. Crea al primo uso."""
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        api_key = _resolve_api_key()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        # Timeout esteso per operazioni lente lato server (es. embed batch su CPU).
        # connect breve, read lungo (600s = 10 min) per non andare in timeout su
        # call /embed o /vector/upsert con tanti dati.
        _client = httpx.Client(
            base_url=_resolve_base_url(),
            timeout=httpx.Timeout(connect=15.0, read=600.0, write=120.0, pool=15.0),
            headers=headers,
        )
        return _client


def request_json(
    method: str,
    path: str,
    *,
    json_body: dict | None = None,
    params: dict | None = None,
) -> Any:
    """Esegue HTTP request con retry. Ritorna JSON parsato."""
    client = get_client()
    last_err: Exception | None = None
    for attempt in range(SERVER.client_max_retries + 1):
        try:
            resp = client.request(
                method, path, json=json_body, params=params,
            )
            if resp.status_code >= 500:
                last_err = httpx.HTTPStatusError(
                    f"Server error {resp.status_code}", request=resp.request, response=resp,
                )
                time.sleep(0.3 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            last_err = e
            time.sleep(0.3 * (attempt + 1))
        except httpx.HTTPStatusError as e:
            # 4xx: non retry-abile
            raise RuntimeError(
                f"Server returned {e.response.status_code}: {e.response.text[:200]}"
            ) from e
    raise RuntimeError(f"Request failed after retries: {last_err}") from last_err


def stream_sse(
    method: str,
    path: str,
    *,
    json_body: dict | None = None,
) -> Iterator[dict]:
    """Generator che yielda dict parsati da eventi SSE `data: {...}\\n\\n`."""
    client = get_client()
    with client.stream(method, path, json=json_body) as resp:
        if resp.status_code >= 400:
            raise RuntimeError(
                f"SSE stream HTTP {resp.status_code}: {resp.read()[:200]!r}"
            )
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload:
                continue
            try:
                yield json.loads(payload)
            except Exception:
                # Skip malformed
                continue
