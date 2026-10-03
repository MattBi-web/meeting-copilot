"""Memory extractor — Fase 15.2.

A fine call, LLM legge la trascrizione completa + le note utente + i suggerimenti
generati, ed estrae "fatti memorabili" da persistere come MemoryRecord.

Tipi di facts estratti:
- semantic: fatti permanenti sul cliente/progetto
  ("Cliente Pharma X usa Salesforce", "Decision maker: Carlo Rossi CTO")
- episodic: eventi specifici di questa call
  ("Il 15/5 hanno richiesto demo per fine mese", "Hanno rifiutato pricing standard")

Ogni fact ha una `importance` (0..1) e uno `scope` (di solito = sezione KB attiva).
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

try:
    import ollama  # noqa: F401
except ImportError:
    ollama = None  # type: ignore[assignment]

from ..config import LLM, MEMORY
from .store import MemoryRecord


_SYSTEM_PROMPT = """Sei un estrattore di memorie da videochiamate professionali.
Leggi una trascrizione di call (consulenza/sales/tech), eventuali note prese
dall'utente, e suggerimenti generati durante la call. Estrai SOLO i fatti che
varrebbe la pena ricordare nelle prossime call con lo stesso cliente/progetto.

═══ TIPI DI MEMORIE ═══

**semantic** — fatti persistenti che varranno anche tra 3 mesi:
- "Cliente X usa Salesforce + SAP"
- "Decision maker: Carlo Rossi (CTO), Maria Bianchi (CFO)"
- "Budget annuale tech: 2M€"
- "Sistema legacy: AS400 in dismissione"

**episodic** — eventi specifici di QUESTA call, contestuali nel tempo:
- "Richiesta demo entro 30 maggio"
- "Obiezione: latenza percepita > 2s inaccettabile"
- "Action item: invio pricing custom entro venerdì"

═══ REGOLE STRINGENTI ═══

1. **NON inventare**. Estrai SOLO quello che è ESPLICITAMENTE nella trascrizione/note.
2. **Importance** (0..1):
   - 0.9-1.0: decision maker names, budget, deadline critiche
   - 0.6-0.8: obiezioni, preferenze, action items
   - 0.3-0.5: contesto generico, note di stile
3. **Compatta**: ogni memoria max 25 parole. Una memoria = un fatto.
4. **Lingua**: stessa della trascrizione.
5. **Niente duplicati**: se due fatti dicono la stessa cosa, tieni il più specifico.

═══ OUTPUT — JSON valido, niente testo extra ═══

{
  "memories": [
    {
      "text": "fact compatto (max 25 parole)",
      "kind": "semantic" | "episodic",
      "importance": 0.0..1.0
    },
    ...
  ]
}

Se la call non ha materiale ricordabile (small talk, problemi tecnici), restituisci:
{"memories": []}
"""


def _parse_json_loose(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", s)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}


def extract_memories_from_session(
    session,                          # Session (duck-typed)
    scope: str,
    *,
    model: str | None = None,
) -> list[MemoryRecord]:
    """Estrae MemoryRecord dalla session corrente.

    Args:
        session: oggetto Session (con .transcript_lines, .notes, .suggestions)
        scope: target scope (es. "acme", "cliente_pharma_x") — di solito = sezione KB attiva
        model: LLM da usare (default LLM.composer_model)

    Returns:
        Lista di MemoryRecord (senza embedding — chi salva lo aggiunge).
    """
    if not MEMORY.enabled:
        return []
    model = model or LLM.composer_model

    transcript = "\n".join(
        f"[{t.timestamp}] {t.text}"
        for t in (session.transcript_lines or [])
    )
    notes = (session.notes or "").strip()
    suggestions = []
    for s in (session.suggestions or []):
        suggestions.append(f"- ({s.category}) {s.text[:200]}")
    suggestions_str = "\n".join(suggestions) if suggestions else "(nessun suggerimento)"

    if not transcript and not notes:
        return []  # call vuota

    # Cap input per il prompt context
    if len(transcript) > 6000:
        transcript = transcript[-6000:]  # ultime righe sono più importanti tipicamente

    user_msg = (
        f"## Trascrizione\n{transcript or '(vuota)'}\n\n"
        f"## Note utente\n{notes or '(nessuna)'}\n\n"
        f"## Suggerimenti generati durante la call\n{suggestions_str}\n\n"
        f"Estrai le memorie in JSON."
    )

    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            format="json",
            options={
                "temperature": 0.2,
                "num_predict": 800,
            },
            keep_alive=LLM.ollama_keep_alive,
            stream=False,
        )
        raw = (resp.get("message", {}).get("content") or "").strip()
    except Exception as e:
        print(f"[memory-extractor] LLM error: {e}")
        return []

    parsed = _parse_json_loose(raw)
    items = parsed.get("memories") or []
    if not isinstance(items, list):
        return []

    out: list[MemoryRecord] = []
    now = time.time()
    for item in items[:MEMORY.max_facts_per_session]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        kind = item.get("kind", "semantic")
        if kind not in ("semantic", "episodic"):
            kind = "semantic"
        try:
            importance = float(item.get("importance", 0.5))
        except Exception:
            importance = 0.5
        importance = max(0.0, min(1.0, importance))

        out.append(MemoryRecord(
            id=uuid.uuid4().hex[:16],
            scope=scope,
            text=text[:500],
            kind=kind,
            importance=importance,
            created_at=now,
            session_id=getattr(session, "id", ""),
            metadata={"raw_session": getattr(session, "id", "")},
        ))

    print(f"[memory-extractor] estratte {len(out)} memorie da session {getattr(session, 'id', '?')}")
    return out
