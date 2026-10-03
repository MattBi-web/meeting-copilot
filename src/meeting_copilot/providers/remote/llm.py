"""Remote LLMProvider — Fase 12.

Supporta sia complete() sincrono che complete_stream() via SSE.
"""
from __future__ import annotations

from collections.abc import Callable

from ..base import LLMMessage
from ._client import request_json, stream_sse


def _messages_to_payload(messages: list[LLMMessage]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


class RemoteLLMProvider:
    """Implementa LLMProvider via HTTP POST /llm/chat + /llm/chat/stream (SSE)."""

    def warmup(self, model: str) -> None:
        # No-op lato client: il server scalda i suoi modelli al primo /llm/chat.
        # Volendo si potrebbe pingare con una mini-call ma evitiamo overhead.
        return None

    def complete(
        self,
        model: str,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.4,
        max_tokens: int = 256,
        format_json: bool = False,
    ) -> str:
        payload = {
            "model": model,
            "messages": _messages_to_payload(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "format_json": format_json,
        }
        data = request_json("POST", "/llm/chat", json_body=payload)
        return data.get("text", "")

    def complete_stream(
        self,
        model: str,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.4,
        max_tokens: int = 256,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        payload = {
            "model": model,
            "messages": _messages_to_payload(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        collected: list[str] = []
        for event in stream_sse("POST", "/llm/chat/stream", json_body=payload):
            if event.get("done"):
                # final event include full text — preferiamo il nostro accumulator
                return event.get("text") or "".join(collected)
            piece = event.get("token", "")
            if piece:
                collected.append(piece)
                if on_token is not None:
                    try:
                        on_token(piece)
                    except Exception:
                        pass
        return "".join(collected)
