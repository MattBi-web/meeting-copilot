"""Cloud Burst — Fase 18+19.

Quando l'utente preme la hotkey di burst (Cmd+Shift+B), un modello cloud
premium con reasoning viene invocato con FULL context: trascrizione recente,
chunks KB rilevanti, memorie correlate, hint sul tipo di query.

Provider supportati:
  - anthropic   → claude-opus-4-6 / claude-sonnet-4-6 con thinking (Fase 19)
  - openai      → gpt-4o / o3 / o3-mini (reasoning_effort)
  - deepseek    → deepseek-reasoner (R1)

L'output è streaming token-by-token come il composer locale, così l'UI può
mostrarlo nello stesso overlay.

Privacy: la trascrizione + i chunks lasciano il device. L'utente deve aver
attivato CLOUD_BURST.enabled in config. La UI mostra un badge "☁️ CLOUD".
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from typing import Any

from ..config import CLOUD_BURST


def _resolve_api_key() -> str:
    key = os.environ.get(CLOUD_BURST.api_key_env, "")
    if not key:
        raise RuntimeError(
            f"Cloud Burst: env var {CLOUD_BURST.api_key_env} non impostata. "
            f"Imposta la API key del tuo provider."
        )
    return key


# ──────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """Sei un copilot AI per call B2B high-stakes. L'utente è IN una
videochiamata RIGHT NOW e ti ha invocato manualmente perché serve una risposta
profonda, ben ragionata, su una domanda complessa.

Ti vengono dati:
1. CONTESTO: ultime righe della trascrizione (chi ha detto cosa)
2. KB_CHUNKS: estratti dalla knowledge base attiva (può essere vuoto)
3. MEMORIES: memorie da call precedenti sullo stesso scope (può essere vuoto)
4. QUERY: la domanda/topic da affrontare

REGOLE STRINGENTI:
- Se KB_CHUNKS contiene info specifiche (numeri, nomi, processi), USALI come fonte primaria.
- Se KB_CHUNKS è vuoto o poco rilevante, usa la tua conoscenza GENERAL ma SEGNALA chiaramente
  "[non da KB]" sui passaggi che non sono ancorati alla KB.
- Se la domanda è una DECISIONE o un'OBIEZIONE, struttura: (1) sintesi, (2) pro/contro,
  (3) opzione consigliata con motivo, (4) cosa dire NOW.
- Lingua: rispondi nella stessa lingua della query (italiano o inglese).
- Lunghezza: 4-8 righe per UI overlay, max ~250 parole. L'utente DEVE poterla leggere
  in 10 secondi mentre parla in call.
- Niente preamboli ("Certo, ecco la risposta..."): vai dritto al punto.
"""


# ──────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────
def _build_user_message(
    query: str,
    transcript_lines: list[str],
    kb_chunks: list,
    memories: list[str],
) -> str:
    parts: list[str] = []

    if transcript_lines:
        ctx = "\n".join(transcript_lines[-12:])
        parts.append(f"=== CONTESTO (ultime righe call) ===\n{ctx}")

    if kb_chunks:
        chunks_str = "\n---\n".join(
            f"[{i+1}] (src: {getattr(c, 'source', '?')}, score: {getattr(c, 'score', 0.0):.2f})\n"
            f"{(getattr(c, 'text', '') or '')[:600]}"
            for i, c in enumerate(kb_chunks[:6])
        )
        parts.append(f"=== KB_CHUNKS ({len(kb_chunks)} totali, top 6) ===\n{chunks_str}")
    else:
        parts.append("=== KB_CHUNKS ===\n(nessun chunk rilevante recuperato)")

    if memories:
        mem_str = "\n".join(f"- {m}" for m in memories[:6])
        parts.append(f"=== MEMORIES (call precedenti) ===\n{mem_str}")

    parts.append(f"=== QUERY ===\n{query}")
    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────
# Anthropic
# ──────────────────────────────────────────────────────────────
def _stream_anthropic(prompt: str, on_token: Callable[[str], None] | None) -> str:
    """Chiamata Anthropic Messages API con SSE + thinking opzionale (Fase 19)."""
    try:
        import httpx
    except ImportError as e:
        raise RuntimeError("httpx non installato") from e

    # Fase 23: budget enforcement pre-call
    try:
        from ..budget import enforce_budget
        from ..budget.tracker import BudgetExceeded
        enforce_budget("anthropic")
    except BudgetExceeded:
        raise
    except Exception:
        pass

    headers = {
        "x-api-key": _resolve_api_key(),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # Fase 19: extended thinking (reasoning mode).
    # Anthropic richiede max_tokens > thinking.budget_tokens (max_tokens
    # include sia thinking sia output finale). Calcoliamo total budget.
    if CLOUD_BURST.reasoning_enabled:
        total_max_tokens = CLOUD_BURST.reasoning_budget_tokens + CLOUD_BURST.max_output_tokens
    else:
        total_max_tokens = CLOUD_BURST.max_output_tokens

    # Prepend session prompt (meta-istruzioni utente)
    try:
        from ..session_prompt import prepend_to_system
        system_with_session = prepend_to_system(_SYSTEM_PROMPT)
    except Exception:
        system_with_session = _SYSTEM_PROMPT

    body: dict[str, Any] = {
        "model": CLOUD_BURST.model,
        "max_tokens": total_max_tokens,
        "stream": True,
        "system": system_with_session,
        "messages": [{"role": "user", "content": prompt}],
    }

    if CLOUD_BURST.reasoning_enabled:
        body["thinking"] = {
            "type": "enabled",
            "budget_tokens": CLOUD_BURST.reasoning_budget_tokens,
        }

    collected: list[str] = []
    in_tokens = 0
    out_tokens = 0
    with httpx.Client(timeout=CLOUD_BURST.request_timeout_s) as client:
        with client.stream(
            "POST", "https://api.anthropic.com/v1/messages",
            headers=headers, json=body,
        ) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Anthropic API error {resp.status_code}: {resp.read()[:200]!r}"
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
                    usage = (event.get("message", {}) or {}).get("usage", {})
                    in_tokens = int(usage.get("input_tokens", 0)) or in_tokens
                elif ev_type == "message_delta":
                    usage = event.get("usage", {})
                    out_tokens = int(usage.get("output_tokens", 0)) or out_tokens

    # Fase 23: record usage post-stream (cloud burst)
    try:
        from ..budget import record_usage
        record_usage("anthropic", CLOUD_BURST.model, in_tokens, out_tokens,
                     scope="burst")
    except Exception:
        pass

    return "".join(collected)


# ──────────────────────────────────────────────────────────────
# OpenAI
# ──────────────────────────────────────────────────────────────
def _stream_openai(prompt: str, on_token: Callable[[str], None] | None) -> str:
    """Chiamata OpenAI Chat Completions API con stream. Supporta o3 reasoning_effort."""
    try:
        import httpx
    except ImportError as e:
        raise RuntimeError("httpx non installato") from e

    headers = {
        "Authorization": f"Bearer {_resolve_api_key()}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": CLOUD_BURST.model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "max_completion_tokens": CLOUD_BURST.max_output_tokens,
    }
    # o3 / o3-mini supportano reasoning_effort: "low" | "medium" | "high"
    if CLOUD_BURST.reasoning_enabled and CLOUD_BURST.model.startswith(("o3", "o4")):
        body["reasoning_effort"] = "medium"

    collected: list[str] = []
    with httpx.Client(timeout=CLOUD_BURST.request_timeout_s) as client:
        with client.stream(
            "POST", "https://api.openai.com/v1/chat/completions",
            headers=headers, json=body,
        ) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"OpenAI API error {resp.status_code}: {resp.read()[:200]!r}"
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
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                piece = delta.get("content") or ""
                if piece:
                    collected.append(piece)
                    if on_token:
                        try:
                            on_token(piece)
                        except Exception:
                            pass
    return "".join(collected)


# ──────────────────────────────────────────────────────────────
# DeepSeek (R1 reasoner)
# ──────────────────────────────────────────────────────────────
def _stream_deepseek(prompt: str, on_token: Callable[[str], None] | None) -> str:
    """DeepSeek API è OpenAI-compatible. Usa endpoint diverso."""
    try:
        import httpx
    except ImportError as e:
        raise RuntimeError("httpx non installato") from e

    headers = {
        "Authorization": f"Bearer {_resolve_api_key()}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": CLOUD_BURST.model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "max_tokens": CLOUD_BURST.max_output_tokens,
    }

    collected: list[str] = []
    with httpx.Client(timeout=CLOUD_BURST.request_timeout_s) as client:
        with client.stream(
            "POST", "https://api.deepseek.com/v1/chat/completions",
            headers=headers, json=body,
        ) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"DeepSeek API error {resp.status_code}: {resp.read()[:200]!r}"
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
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                piece = delta.get("content") or ""
                if piece:
                    collected.append(piece)
                    if on_token:
                        try:
                            on_token(piece)
                        except Exception:
                            pass
    return "".join(collected)


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────
def cloud_burst(
    query: str,
    transcript_lines: list[str] | None = None,
    kb_chunks: list | None = None,
    memories: list[str] | None = None,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """Esegue una burst call al modello cloud configurato e ritorna il testo completo.

    Streaming via `on_token(piece)` se fornito.

    Raises:
        RuntimeError se CLOUD_BURST.enabled=False, API key mancante, o errore HTTP.
    """
    if not CLOUD_BURST.enabled:
        raise RuntimeError(
            "Cloud Burst disabilitato (CLOUD_BURST.enabled=False). "
            "Attivalo in config.py se vuoi usarlo."
        )
    if not query or not query.strip():
        return ""

    prompt = _build_user_message(
        query=query.strip(),
        transcript_lines=transcript_lines or [],
        kb_chunks=kb_chunks or [],
        memories=memories or [],
    )

    provider = CLOUD_BURST.provider.lower().strip()
    if provider == "anthropic":
        return _stream_anthropic(prompt, on_token)
    if provider == "openai":
        return _stream_openai(prompt, on_token)
    if provider == "deepseek":
        return _stream_deepseek(prompt, on_token)
    raise RuntimeError(f"Cloud Burst: provider sconosciuto '{provider}'")
