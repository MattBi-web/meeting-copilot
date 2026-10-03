"""Estrazione L1 — sintesi strutturata per meeting (~300 token).

Sostituisce il summary testuale di M3 con uno schema JSON che alimenta la
piramide (digest L2 sommano da qui, la UI stream card legge da qui).

Schema L1:
    {"summary": str,                       # 2-4 frasi
     "decisions": [str],
     "action_items": [{"owner": str|null, "text": str}],
     "entities": [str],                    # persone/aziende/prodotti citati
     "open_questions": [str]}

Fake mode (MC_LLM_FAKE=1): L1 sintetico deterministico per test.
Costo reale: ~$0.011/meeting con Haiku (8K in / 500 out).
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger("meeting_copilot.brain.extractor")

_SYSTEM = """Sei l'estrattore post-call di Meeting Copilot.
Ricevi il transcript di una call (self = l'utente, others = interlocutori).
Rispondi SOLO con JSON valido, questo schema esatto:
{"summary": "2-4 frasi in italiano",
 "decisions": ["..."],
 "action_items": [{"owner": "nome o null", "text": "..."}],
 "entities": ["persone/aziende/prodotti citati"],
 "open_questions": ["..."]}
Basati SOLO sul transcript. Liste vuote se non c'è nulla. Non inventare."""

_L1_KEYS = {"summary", "decisions", "action_items", "entities", "open_questions"}


def llm_fake_enabled() -> bool:
    return os.environ.get("MC_LLM_FAKE", "") == "1"


def extract_l1(transcript_entries: list, *, language: str | None = "it") -> dict | None:
    """Transcript → dict L1. None se fallisce (best-effort)."""
    if not transcript_entries:
        return None

    if llm_fake_enabled():
        joined = " ".join(e.text for e in transcript_entries)[:120]
        return {
            "summary": f"[fake L1] Call di {len(transcript_entries)} righe: {joined}…",
            "decisions": ["[fake] decisione di prova"],
            "action_items": [{"owner": None, "text": "[fake] azione di prova"}],
            "entities": ["FakeCorp"],
            "open_questions": ["[fake] domanda aperta?"],
        }

    lines = "\n".join(f"[{e.channel}] {e.text}" for e in transcript_entries)
    lines = lines[-24000:]
    try:
        from ..providers import get_llm_provider
        from ..providers.base import LLMMessage
        from ..config import LLM
        provider = get_llm_provider()
        model = getattr(LLM, "analyzer_model", None) or getattr(
            LLM, "composer_model", "qwen2.5:7b-instruct")
        raw = provider.complete(
            model=model,
            messages=[LLMMessage(role="system", content=_SYSTEM),
                      LLMMessage(role="user", content=lines)],
            temperature=0.1, max_tokens=800, format_json=True,
        )
        l1 = _parse_l1(raw)
        if l1 is None:
            log.warning("L1 JSON non parsabile, fallback summary-only")
            return {"summary": raw[:1500], "decisions": [],
                    "action_items": [], "entities": [], "open_questions": []}
        return l1
    except Exception as e:
        log.warning("extract_l1 fallita: %s", e)
        return None


def _parse_l1(raw: str) -> dict | None:
    try:
        # tollera fence markdown
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            cleaned = cleaned.removeprefix("json").strip()
        data = json.loads(cleaned)
        if not isinstance(data, dict) or "summary" not in data:
            return None
        # normalizza chiavi mancanti
        for k in _L1_KEYS:
            data.setdefault(k, [] if k != "summary" else "")
        return {k: data[k] for k in _L1_KEYS}
    except Exception:
        return None


def render_l1_text(l1: dict) -> str:
    """L1 dict → testo per il dialog del client e sessions_meta.summary."""
    parts = [l1.get("summary", "").strip()]
    if l1.get("decisions"):
        parts.append("\nDecisioni:\n" + "\n".join(f"• {d}" for d in l1["decisions"]))
    if l1.get("action_items"):
        rows = []
        for a in l1["action_items"]:
            owner = f" ({a['owner']})" if a.get("owner") else ""
            rows.append(f"• {a.get('text', '')}{owner}")
        parts.append("\nAction items:\n" + "\n".join(rows))
    if l1.get("open_questions"):
        parts.append("\nDomande aperte:\n" + "\n".join(f"• {q}" for q in l1["open_questions"]))
    return "\n".join(p for p in parts if p).strip()
