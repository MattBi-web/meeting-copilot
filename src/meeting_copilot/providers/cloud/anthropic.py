"""AnthropicLLMProvider — Fase 22.

Implementa LLMProvider Protocol parlando direttamente con Anthropic Messages API.
Supporta sia complete() sync sia complete_stream() via SSE.
Rimappa automaticamente il model string al cloud equivalente via CLOUD config.

Format JSON: passato come stringa "Output STRICT in JSON" nel system prompt
(Anthropic non ha un format=json nativo come Ollama).
"""
from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from typing import Any

from ...config import CLOUD
from ..base import LLMMessage


def _resolve_api_key() -> str:
    key = os.environ.get(CLOUD.api_key_env, "")
    if not key:
        raise RuntimeError(
            f"Cloud LLM: env var {CLOUD.api_key_env} non impostata. "
            f"Imposta la API key del provider {CLOUD.provider}."
        )
    return key


# ─────── Latency #4: HTTP/2 persistent client pooling ───────
# Un singolo httpx.Client modulo-level evita SSL/TCP handshake ad ogni call.
# Anthropic supporta HTTP/2 (multiplexing) → -100-200ms per call.
# httpx.Client è thread-safe per uso sync; il _client_lock protegge solo
# la creazione lazy.
_http_client: Any = None
_client_lock = threading.Lock()


def _get_http_client() -> Any:
    """Ritorna un httpx.Client persistente con HTTP/2 + keep-alive."""
    global _http_client
    if _http_client is not None:
        return _http_client
    with _client_lock:
        if _http_client is not None:
            return _http_client
        import httpx
        try:
            _http_client = httpx.Client(
                http2=True,
                timeout=CLOUD.request_timeout_s,
                limits=httpx.Limits(
                    max_keepalive_connections=8,
                    max_connections=16,
                    keepalive_expiry=60.0,
                ),
                headers={"connection": "keep-alive"},
            )
            print("[anthropic] HTTP/2 persistent client initialized")
        except Exception as e:
            # h2 not installed → fallback HTTP/1.1
            print(f"[anthropic] HTTP/2 unavailable ({e}), using HTTP/1.1")
            _http_client = httpx.Client(
                timeout=CLOUD.request_timeout_s,
                limits=httpx.Limits(
                    max_keepalive_connections=8,
                    max_connections=16,
                    keepalive_expiry=60.0,
                ),
            )
        return _http_client


# ─────── Latency #3: Prompt caching ───────
# Anthropic supporta cache_control sul system prompt: il prefill viene cachato
# server-side per 5 min (ephemeral) → -300-500ms TTFT e -90% costo input tokens
# ripetuti. Minimum cache size: 1024 tok Sonnet, 2048 tok Haiku.
# I nostri system prompt (regole + categoria + session prompt) sono ~1500-3000 tok,
# quindi cachable. Cambia solo il PRIMO chunk (dinamico) tra call consecutive.
def _build_system_with_cache(system_str: str) -> Any:
    """Costruisce il campo 'system' Anthropic come list-of-blocks con cache_control.

    Se il system è vuoto → None. Altrimenti single block cachato (semplice).
    """
    if not system_str:
        return None
    return [
        {
            "type": "text",
            "text": system_str,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _remap_model(model: str) -> str:
    """Rimappa model string locale → cloud equivalente in base alla config."""
    lower = model.lower()
    for kw in CLOUD.analyzer_keywords:
        if kw.lower() in lower:
            return CLOUD.analyzer_remap
    for kw in CLOUD.composer_keywords:
        if kw.lower() in lower:
            return CLOUD.composer_remap
    return CLOUD.default_remap


def _split_messages(messages: list[LLMMessage]) -> tuple[str, list[dict]]:
    """Anthropic vuole `system` come parametro separato e `messages` solo
    user/assistant. Splittiamo qui."""
    system_parts: list[str] = []
    out_msgs: list[dict] = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
        else:
            out_msgs.append({"role": m.role, "content": m.content})
    return ("\n\n".join(system_parts) if system_parts else ""), out_msgs


class AnthropicLLMProvider:
    """LLMProvider che parla con Anthropic Messages API.

    Rimappa il model passato (es. "llama3.2:3b") al cloud equivalente
    (es. "claude-haiku-4-5") secondo CLOUD config.
    """

    def warmup(self, model: str) -> None:
        """No-op: Anthropic non richiede warmup."""
        return None

    # ─────── complete sync ───────
    def complete(
        self,
        model: str,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.4,
        max_tokens: int = 256,
        format_json: bool = False,
    ) -> str:
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError("httpx non installato") from e

        # Fase 23: budget enforcement pre-call
        try:
            from ...budget import enforce_budget
            enforce_budget("anthropic")
        except Exception as e:
            # BudgetExceeded oppure ModuleNotFoundError — propaga il primo, ignora il secondo
            from ...budget.tracker import BudgetExceeded
            if isinstance(e, BudgetExceeded):
                raise

        cloud_model = _remap_model(model)
        system, msgs = _split_messages(messages)
        # Prepend session prompt (meta-istruzioni utente, Fase Session)
        try:
            from ...session_prompt import prepend_to_system
            system = prepend_to_system(system or "")
        except Exception:
            pass

        # JSON mode: aggiungiamo istruzione esplicita nel system prompt
        if format_json:
            extra = ("\n\nIMPORTANTE: rispondi ESCLUSIVAMENTE con JSON valido, "
                     "senza testo extra prima o dopo. Niente markdown wrapper.")
            system = (system + extra) if system else extra.lstrip()

        body: dict[str, Any] = {
            "model": cloud_model,
            "max_tokens": max_tokens,
            "messages": msgs,
            "temperature": temperature,
        }
        # Latency #3: prompt caching su system prompt
        sys_blocks = _build_system_with_cache(system)
        if sys_blocks:
            body["system"] = sys_blocks

        headers = {
            "x-api-key": _resolve_api_key(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        # Latency #4: client HTTP/2 persistent
        client = _get_http_client()
        resp = client.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers, json=body,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Anthropic API error {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()

        # Extract testo dal content blocks (skip thinking)
        content_blocks = data.get("content", [])
        text_parts: list[str] = []
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        # Fase 23: record usage post-call
        try:
            from ...budget import record_usage
            usage = data.get("usage", {})
            in_tok = int(usage.get("input_tokens", 0))
            out_tok = int(usage.get("output_tokens", 0))
            record_usage("anthropic", cloud_model, in_tok, out_tok, scope="complete")
        except Exception:
            pass

        return "".join(text_parts)

    # ─────── complete streaming (SSE) ───────
    def complete_stream(
        self,
        model: str,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.4,
        max_tokens: int = 256,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError("httpx non installato") from e

        # Fase 23: budget enforcement pre-call
        try:
            from ...budget import enforce_budget
            from ...budget.tracker import BudgetExceeded
            enforce_budget("anthropic")
        except BudgetExceeded:
            raise
        except Exception:
            pass

        cloud_model = _remap_model(model)
        system, msgs = _split_messages(messages)
        # Prepend session prompt (meta-istruzioni utente, Fase Session)
        try:
            from ...session_prompt import prepend_to_system
            system = prepend_to_system(system or "")
        except Exception:
            pass

        # Reasoning per composer (riconosciuto dal mapping)
        is_composer = cloud_model == CLOUD.composer_remap
        use_reasoning = is_composer and CLOUD.composer_reasoning_enabled

        # max_tokens deve essere > thinking budget (Anthropic requirement)
        total_max = (
            max_tokens + CLOUD.composer_reasoning_budget if use_reasoning
            else max_tokens
        )

        body: dict[str, Any] = {
            "model": cloud_model,
            "max_tokens": total_max,
            "stream": True,
            "messages": msgs,
            "temperature": temperature,
        }
        # Latency #3: prompt caching su system prompt
        sys_blocks = _build_system_with_cache(system)
        if sys_blocks:
            body["system"] = sys_blocks
        if use_reasoning:
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": CLOUD.composer_reasoning_budget,
            }

        headers = {
            "x-api-key": _resolve_api_key(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        collected: list[str] = []
        in_tokens = 0
        out_tokens = 0
        # Latency #4: client HTTP/2 persistent (riusato across calls)
        client = _get_http_client()
        with client.stream(
            "POST", "https://api.anthropic.com/v1/messages",
            headers=headers, json=body,
        ) as resp:
                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"Anthropic stream HTTP {resp.status_code}: "
                        f"{resp.read()[:200]!r}"
                    )
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        event = json.loads(payload)
                    except Exception:
                        continue
                    ev_type = event.get("type")
                    if ev_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            piece = delta.get("text", "")
                            if piece:
                                collected.append(piece)
                                if on_token:
                                    try:
                                        on_token(piece)
                                    except Exception:
                                        pass
                    elif ev_type == "message_start":
                        # message_start contiene usage iniziale (input_tokens)
                        usage = (event.get("message", {}) or {}).get("usage", {})
                        in_tokens = int(usage.get("input_tokens", 0)) or in_tokens
                    elif ev_type == "message_delta":
                        # message_delta finale ha usage.output_tokens
                        usage = event.get("usage", {})
                        out_tokens = int(usage.get("output_tokens", 0)) or out_tokens

        # Fase 23: record usage post-stream
        try:
            from ...budget import record_usage
            record_usage("anthropic", cloud_model, in_tokens, out_tokens, scope="stream")
        except Exception:
            pass

        return "".join(collected)
