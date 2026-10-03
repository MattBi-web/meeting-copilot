"""Planner agent — Fase 16.1.

Riceve query + meta-info di contesto (sezioni KB attive, presenza memorie,
tipo di intent rilevato dal Watcher) e produce un PLAN strutturato:
una sequenza di step che l'AgentOrchestrator dovrà eseguire.

Filosofia: il modello di pianificazione NON deve fare retrieval né compose.
Decide SOLO la strategia. Latency obiettivo: 200-400ms su llama3.2:3b.

Step disponibili (vocabolario chiuso, validato post-parse):
    - retrieve_kb            → cerca nella knowledge base attiva
    - retrieve_memory        → cerca nella episodic memory cross-call
    - query_expand           → genera 2-3 varianti della query (CRAG-like)
    - evaluate_chunks        → chiama l'Evaluator per giudicare sufficienza
    - compose_grounded       → composer in modalità HIGH/MEDIUM (con KB)
    - compose_fallback       → composer in modalità LOW (LLM knowledge only)
    - stop                   → conclude senza generare risposta

Esempi di plan ragionati:

1. Domanda fattuale tecnica con KB attiva:
   [retrieve_kb, evaluate_chunks, compose_grounded]

2. Domanda fattuale con KB attiva ma query vaga:
   [query_expand, retrieve_kb, evaluate_chunks, compose_grounded]

3. Riferimento a discussione precedente ("come avevamo detto"):
   [retrieve_memory, retrieve_kb, compose_grounded]

4. Obiezione su prezzo senza KB pricing:
   [retrieve_memory, compose_fallback]

5. Smalltalk:
   [stop]
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

try:
    import ollama  # noqa: F401
except ImportError:
    ollama = None  # type: ignore[assignment]

from ..config import LLM, RAG


Step = Literal[
    "retrieve_kb",
    "retrieve_memory",
    "query_expand",
    "evaluate_chunks",
    "compose_grounded",
    "compose_fallback",
    "stop",
]

ALLOWED_STEPS: tuple[Step, ...] = (
    "retrieve_kb",
    "retrieve_memory",
    "query_expand",
    "evaluate_chunks",
    "compose_grounded",
    "compose_fallback",
    "stop",
)


@dataclass(frozen=True)
class Plan:
    """Plan di esecuzione prodotto dal Planner."""
    steps: list[Step]                     # sequenza di step
    reasoning: str                        # breve giustificazione (max 60 parole)
    confidence: float                     # 0..1 — quanto il planner è sicuro
    raw: str = ""                         # output grezzo LLM (debug)
    fallback: bool = False                # True = fallback heuristic (non LLM)


_SYSTEM_PROMPT = """Sei un PLANNER di un sistema RAG agentico in real-time per call sales.
Ricevi: QUERY dell'utente, CONTEXT (sezioni KB attive, presenza memorie, intent rilevato).
Devi decidere UNA SEQUENZA DI STEP da eseguire — niente di più, niente di meno.

VOCABOLARIO STEP (usa SOLO questi nomi, esatti):
- retrieve_kb       — cerca nella knowledge base attiva
- retrieve_memory   — cerca memorie da call passate (scope = sezioni KB)
- query_expand      — genera 2-3 varianti della query (sinonimi/parafrasi)
- evaluate_chunks   — valuta se i chunks retrievati sono sufficienti
- compose_grounded  — genera risposta usando KB chunks (tier HIGH/MEDIUM)
- compose_fallback  — genera risposta SOLO con knowledge LLM (tier LOW, con disclaimer)
- stop              — non rispondere (smalltalk, niente di utile da dire)

REGOLE DI PIANIFICAZIONE:

1. Se KB è ATTIVA e query è fattuale/tecnica → SEMPRE inizia con retrieve_kb (o query_expand→retrieve_kb se query vaga).
2. Se la query fa riferimento a discussioni precedenti ("come dicevo", "ricordi", "ne avevamo parlato") → AGGIUNGI retrieve_memory PRIMA del retrieve_kb.
3. Se è opportuno valutare la qualità dei chunks (query complessa) → evaluate_chunks dopo retrieve.
4. compose_grounded SOLO se almeno un retrieve_* è nel plan.
5. compose_fallback se NESSUN retrieve_* è utile (KB vuota O query fuori scope KB O smalltalk con value).
6. stop SOLO per smalltalk genuino senza azione richiesta.
7. Numero step: minimo 1 (stop), massimo 5. Più step = più latenza.

OUTPUT — JSON valido (e SOLO JSON, niente testo extra):
{
  "steps": ["step1", "step2", ...],
  "reasoning": "breve giustificazione max 60 parole",
  "confidence": 0.0..1.0
}
"""


def _parse_json_loose(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        pass
    match = re.search(r"\{.*\}", s, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}


def _validate_steps(raw_steps) -> list[Step]:
    """Validation + sanitization: filtra solo step ammessi, mantieni ordine."""
    if not isinstance(raw_steps, list):
        return []
    out: list[Step] = []
    for s in raw_steps:
        if not isinstance(s, str):
            continue
        s = s.strip().lower()
        if s in ALLOWED_STEPS:
            out.append(s)  # type: ignore[arg-type]
    return out


def _heuristic_plan(query: str, has_kb: bool, has_memory: bool, intent_category: str) -> Plan:
    """Fallback senza LLM. Logica deterministica basata su regole semplici.

    Usato in 2 casi:
    1. Planner LLM fallisce (timeout, errore, JSON malformato)
    2. Planner disabilitato → usiamo questo come quick path
    """
    q = (query or "").strip().lower()

    # Smalltalk / niente di utile
    if not q or intent_category == "smalltalk":
        return Plan(
            steps=["stop"],
            reasoning="smalltalk o query vuota — niente da fare",
            confidence=0.95,
            fallback=True,
        )

    # Riferimento a memoria
    memory_signals = (
        "come dicevo", "come avevo detto", "ricordi", "ne avevamo parlato",
        "l'altra volta", "ne abbiamo parlato", "come dicevamo",
        "as i mentioned", "as we discussed", "we said", "remember",
    )
    needs_memory = has_memory and any(sig in q for sig in memory_signals)

    # Query vaga (poche parole, no entità) → benefit query_expand
    is_vague = len(q.split()) <= 4 and not any(c.isupper() for c in query)

    # Obiezione → spesso meglio fallback (LLM scrive risposta strutturata)
    if intent_category == "objection":
        steps: list[Step] = []
        if needs_memory:
            steps.append("retrieve_memory")
        if has_kb:
            steps.append("retrieve_kb")
            steps.append("compose_grounded")
        else:
            steps.append("compose_fallback")
        return Plan(
            steps=steps,
            reasoning="obiezione: memoria + KB se disponibile, altrimenti LLM",
            confidence=0.7,
            fallback=True,
        )

    # Default: question / pitch_opportunity / info_request
    steps = []
    if needs_memory:
        steps.append("retrieve_memory")
    if has_kb:
        if is_vague:
            steps.append("query_expand")
        steps.append("retrieve_kb")
        if is_vague:
            steps.append("evaluate_chunks")
        steps.append("compose_grounded")
    else:
        steps.append("compose_fallback")

    return Plan(
        steps=steps,
        reasoning=f"heuristic: kb={has_kb} mem={needs_memory} vague={is_vague}",
        confidence=0.65,
        fallback=True,
    )


class Planner:
    """Planner agent — wrapper sul modello LLM di pianificazione."""

    def __init__(self, model: str | None = None):
        self.model = model or RAG.agentic_planner_model

    def plan(
        self,
        query: str,
        has_kb: bool,
        has_memory: bool,
        intent_category: str = "question",
    ) -> Plan:
        """Produce un Plan a partire da query + context flags.

        Args:
            query: la query/topic (idealmente già extracted dal Watcher).
            has_kb: c'è almeno una sezione KB attiva con chunks?
            has_memory: c'è almeno una memoria nello scope corrente?
            intent_category: categoria del Watcher (question|objection|pitch_opportunity|smalltalk).

        Returns:
            Plan validato. Se LLM fallisce, ritorna heuristic plan.
        """
        # Pre-flight: query vuota → stop
        if not query or not query.strip():
            return Plan(
                steps=["stop"], reasoning="empty query", confidence=1.0, fallback=True
            )

        user_msg = (
            f"QUERY: {query}\n"
            f"CONTEXT:\n"
            f"- knowledge_base_attiva: {has_kb}\n"
            f"- memorie_disponibili: {has_memory}\n"
            f"- intent_category: {intent_category}\n\n"
            f"Pianifica."
        )

        try:
            resp = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                format="json",
                options={
                    "temperature": 0.1,
                    "num_predict": 180,
                },
                keep_alive=LLM.ollama_keep_alive,
                stream=False,
            )
            raw = (resp.get("message", {}).get("content") or "").strip()
        except Exception as e:
            print(f"[planner] LLM error: {e} → fallback heuristic")
            p = _heuristic_plan(query, has_kb, has_memory, intent_category)
            return Plan(
                steps=p.steps, reasoning=f"LLM error: {e} | {p.reasoning}",
                confidence=p.confidence, raw=f"ERROR: {e}", fallback=True,
            )

        parsed = _parse_json_loose(raw)
        steps = _validate_steps(parsed.get("steps"))

        if not steps:
            # JSON malformato o nessun step valido → heuristic
            p = _heuristic_plan(query, has_kb, has_memory, intent_category)
            return Plan(
                steps=p.steps,
                reasoning=f"invalid LLM plan, fallback heuristic | raw={raw[:80]!r}",
                confidence=p.confidence, raw=raw, fallback=True,
            )

        # Sanity check: enforce vincoli minimi
        steps = _enforce_constraints(steps, has_kb)
        # Cap massimo lunghezza
        if len(steps) > RAG.agentic_max_steps:
            steps = steps[: RAG.agentic_max_steps]

        reasoning = (parsed.get("reasoning") or "")[:240]
        confidence = float(parsed.get("confidence", 0.6))
        confidence = max(0.0, min(1.0, confidence))

        return Plan(
            steps=steps,
            reasoning=reasoning,
            confidence=confidence,
            raw=raw,
            fallback=False,
        )


def _enforce_constraints(steps: list[Step], has_kb: bool) -> list[Step]:
    """Post-validation: applica vincoli di consistenza al plan.

    Regole:
    - se compose_grounded c'è ma nessun retrieve_*, rimpiazza con compose_fallback
    - se ci sono SOLO retrieve_* senza compose_*, aggiungi compose_grounded o
      compose_fallback in coda
    - se has_kb=False, rimuovi retrieve_kb / evaluate_chunks (inutili)
    - dedup conservativo (mantiene primo)
    """
    if not has_kb:
        steps = [s for s in steps if s not in ("retrieve_kb", "evaluate_chunks")]

    # Dedup mantenendo ordine
    seen = set()
    deduped: list[Step] = []
    for s in steps:
        if s not in seen:
            deduped.append(s)
            seen.add(s)
    steps = deduped

    has_retrieve = any(s.startswith("retrieve_") for s in steps)
    has_compose = any(s.startswith("compose_") for s in steps) or "stop" in steps

    # compose_grounded senza retrieve → declassa a fallback
    if "compose_grounded" in steps and not has_retrieve:
        idx = steps.index("compose_grounded")
        steps[idx] = "compose_fallback"

    # Plan senza compose finale → aggiungi
    if not has_compose:
        if has_retrieve:
            steps.append("compose_grounded")
        else:
            steps.append("compose_fallback")

    return steps
