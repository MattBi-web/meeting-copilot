"""Coordinator — chat off-call sulla piramide (v1.1b, DESIGN_PRODUCT_V1_1 §4).

Router-first, tre scope:
    everywhere      → router Haiku legge l'index card L3 e decide quali stream
                      servono (o risponde dai soli digest per domande aggregate)
    stream:<slug>   → salta il router, contesto = quel digest + drill-down
    meeting:<sid>   → contesto = L1 di quella call

Contesto per la risposta (in ordine di economicità):
    L3 index card (sempre) → L2 digest degli stream scelti → L1 degli ultimi
    meeting (se drill) → chunk KB via retrieval esistente (se drill).

Costi attesi: $0.02-0.04 a domanda cross-stream (router Haiku ~2K tok +
risposta su ~8-10K tok). TODO ottimizzazione: prompt caching Anthropic su
L3+L2 (richiede supporto cache_control nel provider cloud — non ancora
implementato lì).

Fake mode (MC_LLM_FAKE=1): router deterministico + risposta sintetica.
"""
from __future__ import annotations

import json
import logging

from .extractor import llm_fake_enabled
from .store import BrainStore

log = logging.getLogger("meeting_copilot.brain.coordinator")

MAX_HISTORY = 6
DRILL_MEETINGS_PER_STREAM = 5
DRILL_CHUNKS = 4

_ROUTER_SYSTEM = """Sei il router del Coordinator di Meeting Copilot.
Ricevi l'INDEX degli stream di lavoro dell'utente e una domanda.
Rispondi SOLO con JSON:
{"strategy": "digests_only" | "drill",
 "streams": ["slug", ...]}

Regole:
- domanda aggregata/panoramica su più progetti → "digests_only" + tutti gli stream rilevanti
- domanda puntuale su fatti/dettagli → "drill" + gli 1-2 stream più pertinenti
- se nessuno stream è pertinente → "digests_only" con streams []"""

_ANSWER_SYSTEM = """Sei il Coordinator di Meeting Copilot: l'assistente che ha
visione di tutti gli stream di lavoro dell'utente (meeting, decisioni, azioni).
Rispondi in italiano, conciso e operativo.
REGOLE:
- Basati SOLO sul contesto fornito (index, digest, sintesi meeting, estratti).
- Cita sempre da dove viene l'informazione: (stream X) o (meeting del GG/MM) o (fonte: file).
- Se il contesto non basta, dillo esplicitamente e suggerisci dove guardare.
- Le informazioni più recenti prevalgono su quelle più vecchie."""


def _llm_complete(system: str, user: str, *, max_tokens: int = 900,
                  temperature: float = 0.3, format_json: bool = False) -> str:
    from ..providers import get_llm_provider
    from ..providers.base import LLMMessage
    from ..config import LLM
    model = getattr(LLM, "analyzer_model", None) or getattr(
        LLM, "composer_model", "qwen2.5:7b-instruct")
    return get_llm_provider().complete(
        model=model,
        messages=[LLMMessage(role="system", content=system),
                  LLMMessage(role="user", content=user)],
        temperature=temperature, max_tokens=max_tokens,
        format_json=format_json,
    )


def _route(card: str, question: str, all_streams: list[str]) -> dict:
    if llm_fake_enabled():
        return {"strategy": "drill", "streams": all_streams[:2]}
    try:
        raw = _llm_complete(_ROUTER_SYSTEM,
                            f"INDEX:\n{card}\n\nDomanda: {question}",
                            max_tokens=200, temperature=0.0, format_json=True)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1].removeprefix("json").strip()
        plan = json.loads(cleaned)
        streams = [s for s in plan.get("streams", []) if s in all_streams]
        strategy = plan.get("strategy", "digests_only")
        return {"strategy": strategy, "streams": streams}
    except Exception as e:
        log.warning("router fallito (%s) → fallback digests_only su tutto", e)
        return {"strategy": "digests_only", "streams": all_streams}


def _kb_chunks(question: str, streams: list[str], user_id: str) -> list[dict]:
    """Drill-down nei chunk KB degli stream scelti (retrieval esistente)."""
    try:
        from ..kb.manager import KnowledgeManager
        # NB: per utenti autenticati gli scope sono prefissati (tenant);
        # il Coordinator in dashboard gira oggi single-user → slug puri.
        hits = KnowledgeManager.search(question, sections=streams,
                                       top_k=DRILL_CHUNKS)
        return [{"text": h.text[:400], "source": h.source,
                 "section": h.section} for h in hits]
    except Exception as e:
        log.debug("kb drill saltato: %s", e)
        return []


def ask(question: str, *, user_id: str, scope: str = "everywhere",
        history: list[dict] | None = None) -> dict:
    """Risponde a una domanda. Ritorna:
    {"answer", "strategy", "streams", "sources": [..], "context_tokens_est"}
    """
    store = BrainStore.instance()
    digests = [d for d in store.get_digests(user_id)
               if d["stream"] != "_unassigned"]
    card = store.get_index_card(user_id) or "(index non ancora generata — esegui il consolidamento)"
    all_streams = [d["stream"] for d in digests]

    # ── scope → piano ──
    if scope.startswith("stream:"):
        slug = scope.split(":", 1)[1]
        plan = {"strategy": "drill", "streams": [slug] if slug in all_streams else []}
    elif scope.startswith("meeting:"):
        plan = {"strategy": "meeting", "streams": []}
    else:
        plan = _route(card, question, all_streams)

    # ── costruzione contesto ──
    blocks: list[str] = [f"=== INDEX ===\n{card}"]
    sources: list[str] = []

    chosen = plan["streams"] or all_streams[:6]
    digest_map = {d["stream"]: d for d in digests}
    for s in chosen:
        d = digest_map.get(s)
        if d and d["digest"]:
            blocks.append(f"=== DIGEST stream «{s}» ({d['n_meetings']} meeting) ===\n{d['digest']}")
            sources.append(f"digest:{s}")

    if plan["strategy"] == "meeting" and scope.startswith("meeting:"):
        sid = scope.split(":", 1)[1]
        for s in all_streams:
            for m in store.meetings_for_stream(user_id, s, limit=100):
                if m["_session_id"] == sid:
                    blocks.append(f"=== MEETING {sid} (stream {s}) ===\n"
                                  f"{json.dumps(m, ensure_ascii=False)[:3000]}")
                    sources.append(f"meeting:{sid}")

    if plan["strategy"] == "drill":
        for s in plan["streams"]:
            for m in store.meetings_for_stream(user_id, s,
                                               limit=DRILL_MEETINGS_PER_STREAM):
                import time as _t
                day = _t.strftime("%d/%m", _t.localtime(m["_ts"]))
                blocks.append(f"=== MEETING {day} (stream {s}) ===\n"
                              f"{m.get('summary', '')}\n"
                              f"decisioni: {m.get('decisions', [])}\n"
                              f"azioni: {m.get('action_items', [])}")
                sources.append(f"meeting:{s}:{day}")
        for c in _kb_chunks(question, plan["streams"], user_id):
            blocks.append(f"=== ESTRATTO ({c['section']} / {c['source']}) ===\n{c['text']}")
            sources.append(f"kb:{c['source']}")

    context = "\n\n".join(blocks)
    context = context[-40000:]  # hard cap ~10K token

    # ── risposta ──
    if llm_fake_enabled():
        answer = (f"[fake coordinator] strategia={plan['strategy']}, "
                  f"stream={plan['streams']}, blocchi contesto={len(blocks)}, "
                  f"fonti={len(sources)}. Domanda: {question[:60]}")
    else:
        hist_txt = ""
        for h in (history or [])[-MAX_HISTORY:]:
            hist_txt += f"\n{h.get('role', 'user')}: {h.get('text', '')[:400]}"
        user_msg = (f"CONTESTO:\n{context}\n\n"
                    + (f"CONVERSAZIONE PRECEDENTE:{hist_txt}\n\n" if hist_txt else "")
                    + f"DOMANDA: {question}")
        try:
            answer = _llm_complete(_ANSWER_SYSTEM, user_msg, max_tokens=900)
        except Exception as e:
            log.error("coordinator answer fallita: %s", e)
            answer = f"Errore nel generare la risposta: {e}"

    return {
        "answer": answer,
        "strategy": plan["strategy"],
        "streams": plan["streams"],
        "sources": sources[:12],
        "context_tokens_est": len(context) // 4,
    }
