"""AgentOrchestrator — Fase 16.2.

Esegue un Plan prodotto dal Planner come state-machine deterministica.

State machine:
    AgentState (mutable):
      - query: str
      - effective_query: str            # eventualmente rewritten / expanded
      - sections: list[str]             # KB attive
      - hits: list[SearchHit]           # chunks recuperati (KB)
      - memory_context: str             # memorie cross-call formattate
      - query_variants: list[str]       # se query_expand è stato eseguito
      - evaluation: EvaluationResult|None
      - grounding_tier: str             # "high" | "medium" | "low"
      - response_text: str              # accumulator dello streaming composer
      - composed: bool                  # True dopo compose_*
      - logs: list[str]                 # trace step-by-step
      - aborted: bool                   # True se uno step ha forzato stop

Lo step composer è l'unico che STREAMA via on_token callback. Gli altri step
sono sincroni e popolano lo state.

Backward compat: l'AgentOrchestrator NON sostituisce l'Orchestrator legacy.
È invocato come ramo alternativo dal Watcher (e in futuro dall'Orchestrator)
quando RAG.agentic_mode_enabled = True.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from ..config import LLM, MEMORY, RAG
from ..kb.manager import KnowledgeManager
from ..kb.retriever import SearchHit
from .composer import Composer
from .planner import Plan, Planner


# ──────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────
@dataclass
class AgentState:
    query: str
    sections: list[str]
    category: str = "question"
    effective_query: str = ""
    hits: list[SearchHit] = field(default_factory=list)
    memory_context: str = ""
    query_variants: list[str] = field(default_factory=list)
    evaluation: object | None = None  # EvaluationResult
    grounding_tier: str = "medium"
    response_text: str = ""
    composed: bool = False
    logs: list[str] = field(default_factory=list)
    aborted: bool = False
    transcript_lines: list[str] = field(default_factory=list)


@dataclass
class AgentRunResult:
    """Risultato finale di una run dell'AgentOrchestrator."""
    response_text: str
    hits: list[SearchHit]
    plan: Plan
    state: AgentState
    elapsed_ms: float
    success: bool                       # True se ha generato output o stop motivato


# ──────────────────────────────────────────────────────────────
# AgentOrchestrator
# ──────────────────────────────────────────────────────────────
class AgentOrchestrator:
    """Esegue un Plan stepwise. Thread-safe per istanze diverse (no shared state)."""

    def __init__(
        self,
        on_token: Callable[[str], None] | None = None,
        on_step: Callable[[str, str], None] | None = None,    # (step_name, status_text)
        composer: Composer | None = None,
        planner: Planner | None = None,
    ):
        self._on_token = on_token or (lambda _t: None)
        self._on_step = on_step or (lambda _s, _t: None)
        self._composer = composer or Composer()
        self._planner = planner or Planner()

    # ─────── Entry point ───────
    def run(
        self,
        query: str,
        sections: list[str],
        category: str = "question",
        transcript_lines: list[str] | None = None,
        has_memory_signal: bool | None = None,
    ) -> AgentRunResult:
        """Genera un plan e lo esegue stepwise.

        Args:
            query: la query da elaborare.
            sections: scope KB attivo (sezioni).
            category: intent dal Watcher (question|objection|pitch_opportunity|smalltalk).
            transcript_lines: ultime righe della trascrizione (per il composer).
            has_memory_signal: hint che la query fa riferimento a memorie passate.
                               Se None, lasciamo che il planner decida.

        Returns:
            AgentRunResult con tutti i dati per logging/UI.
        """
        t0 = time.monotonic()
        state = AgentState(
            query=query.strip(),
            effective_query=query.strip(),
            sections=list(sections or []),
            category=category,
            transcript_lines=list(transcript_lines or []),
        )

        has_kb = bool(state.sections)
        has_memory = bool(has_memory_signal) if has_memory_signal is not None \
            else self._guess_memory_relevance(state.query)

        # 1. Plan
        self._on_step("plan", "● Pianificando…")
        plan = self._planner.plan(
            query=state.query,
            has_kb=has_kb,
            has_memory=has_memory,
            intent_category=category,
        )
        state.logs.append(
            f"plan: {plan.steps} (conf={plan.confidence:.2f}, fallback={plan.fallback})"
        )
        print(f"[agent] plan={plan.steps} conf={plan.confidence:.2f} fallback={plan.fallback}")

        # 2. Esegui ogni step
        for step in plan.steps:
            if state.aborted:
                break
            self._dispatch_step(step, state)

        elapsed_ms = (time.monotonic() - t0) * 1000
        success = state.composed or "stop" in plan.steps
        print(f"[agent] done in {elapsed_ms:.0f}ms — composed={state.composed} aborted={state.aborted}")

        return AgentRunResult(
            response_text=state.response_text,
            hits=state.hits,
            plan=plan,
            state=state,
            elapsed_ms=elapsed_ms,
            success=success,
        )

    # ─────── Dispatcher ───────
    def _dispatch_step(self, step: str, state: AgentState) -> None:
        try:
            if step == "retrieve_kb":
                self._step_retrieve_kb(state)
            elif step == "retrieve_memory":
                self._step_retrieve_memory(state)
            elif step == "query_expand":
                self._step_query_expand(state)
            elif step == "evaluate_chunks":
                self._step_evaluate_chunks(state)
            elif step == "compose_grounded":
                self._step_compose(state, fallback_mode=False)
            elif step == "compose_fallback":
                self._step_compose(state, fallback_mode=True)
            elif step == "stop":
                self._step_stop(state)
            else:
                state.logs.append(f"unknown step skipped: {step}")
        except Exception as e:
            state.logs.append(f"{step} error: {e}")
            print(f"[agent] step '{step}' error: {e}")

    # ─────── Step implementations ───────
    def _step_retrieve_kb(self, state: AgentState) -> None:
        if not state.sections:
            state.logs.append("retrieve_kb: no sections, skip")
            return
        self._on_step("retrieve_kb", "● Cerco nella knowledge base…")

        # Se abbiamo query_variants precedenti dal planner step, le includiamo
        # facendo merge dei risultati. Senza variants, usa solo state.query.
        queries = [state.effective_query]
        if state.query_variants:
            queries.extend(state.query_variants)

        try:
            from ..config import RAG as _RAG
            from ..kb.retriever import _compute_grounding, search

            all_hits: dict[tuple, SearchHit] = {}
            for q in queries:
                if not q.strip():
                    continue
                # Skip rewriting solo per le variants (sono già query, non trascrizioni)
                hits = search(
                    q, sections=state.sections, top_k=_RAG.final_top_k,
                    skip_rewriting=(q != state.effective_query),
                )
                for h in hits:
                    key = (h.section, h.source, h.text[:100])
                    if key not in all_hits or h.score > all_hits[key].score:
                        all_hits[key] = h
            merged = sorted(all_hits.values(), key=lambda h: h.score, reverse=True)
            state.hits = merged[: _RAG.final_top_k]
            meta = _compute_grounding(state.hits)
            state.grounding_tier = (
                meta.tier if _RAG.confidence_gating_enabled else "medium"
            )
            state.logs.append(
                f"retrieve_kb: {len(state.hits)} hits, tier={state.grounding_tier} "
                f"(queries={len(queries)})"
            )
        except Exception as e:
            state.logs.append(f"retrieve_kb error: {e}")
            print(f"[agent] retrieve_kb error: {e}")

    def _step_retrieve_memory(self, state: AgentState) -> None:
        if not MEMORY.enabled:
            state.logs.append("retrieve_memory: disabled, skip")
            return
        self._on_step("retrieve_memory", "● Cerco memorie da call passate…")

        try:
            from ..memory import MemoryManager
            scopes = state.sections or [MEMORY.default_scope]
            memories = MemoryManager.search(
                state.effective_query, scopes, top_k=MEMORY.max_memories_per_query
            )
            if memories:
                state.memory_context = MemoryManager.format_as_context(memories)
                state.logs.append(f"retrieve_memory: {len(memories)} memorie trovate")
                print(f"[agent] memory: {len(memories)} found")
            else:
                state.logs.append("retrieve_memory: 0 trovate")
        except Exception as e:
            state.logs.append(f"retrieve_memory error: {e}")
            print(f"[agent] retrieve_memory error: {e}")

    def _step_query_expand(self, state: AgentState) -> None:
        """Usa l'Evaluator per generare variants ANCHE senza retrieve precedente.
        Strategia: facciamo un retrieve "lite" di sondaggio, poi chiediamo
        variants se il primo retrieval è povero, OPPURE generiamo
        sinonimi/parafrasi via mini-LLM call ad-hoc.

        Per evitare doppio retrieve, qui usiamo direttamente un LLM call
        veloce con prompt dedicato (3 varianti max). Latency ~300ms.
        """
        self._on_step("query_expand", "● Espando la query…")
        try:
            variants = _generate_query_variants(state.effective_query)
            state.query_variants = variants
            state.logs.append(f"query_expand: {variants}")
        except Exception as e:
            state.logs.append(f"query_expand error: {e}")
            print(f"[agent] query_expand error: {e}")

    def _step_evaluate_chunks(self, state: AgentState) -> None:
        if not state.hits:
            state.logs.append("evaluate_chunks: no hits, skip")
            return
        self._on_step("evaluate_chunks", "● Valuto i risultati…")
        try:
            from .evaluator import Evaluator
            ev = Evaluator()
            result = ev.evaluate(state.effective_query, state.hits)
            state.evaluation = result
            state.logs.append(
                f"evaluate_chunks: verdict={result.verdict} conf={result.confidence:.2f}"
            )
            # Se "partial" e abbiamo nuove variants suggerite, le iniettiamo
            # nello state — un retrieve_kb successivo (se presente nel plan) le userà
            if result.verdict == "partial" and result.suggested_query_variants:
                state.query_variants = (
                    state.query_variants + result.suggested_query_variants
                )[:3]
                state.logs.append(
                    f"evaluate_chunks: aggiunte {len(result.suggested_query_variants)} variants"
                )
            # Se "insufficient" → forziamo grounding tier low (composer userà fallback)
            if result.verdict == "insufficient":
                state.grounding_tier = "low"
        except Exception as e:
            state.logs.append(f"evaluate_chunks error: {e}")
            print(f"[agent] evaluate_chunks error: {e}")

    def _step_compose(self, state: AgentState, *, fallback_mode: bool) -> None:
        """Composer step con streaming.

        Args:
            fallback_mode: True = compose_fallback (tier LOW, no KB),
                           False = compose_grounded (usa state.hits e tier corrente).
        """
        if fallback_mode:
            self._on_step("compose_fallback", "● Genero risposta (general knowledge)…")
            tier = "low"
            chunks_for_composer = []
        else:
            self._on_step("compose_grounded", "● Genero risposta (KB-grounded)…")
            tier = state.grounding_tier or "medium"
            chunks_for_composer = state.hits

        # Memory come contesto extra (se presente)
        tx = list(state.transcript_lines)
        if state.memory_context:
            tx = [state.memory_context] + tx

        def on_piece(piece: str) -> None:
            state.response_text += piece
            self._on_token(piece)

        try:
            full_text = self._composer.compose_stream(
                category=state.category,
                query=state.effective_query,
                chunks=chunks_for_composer,
                transcript_lines=tx,
                on_token=on_piece,
                grounding_tier=tier,
            )
            state.response_text = full_text  # truth source: ritornato completo
            state.composed = True
            state.logs.append(
                f"compose_{'fallback' if fallback_mode else 'grounded'}: "
                f"{len(full_text)} chars, tier={tier}"
            )
        except Exception as e:
            state.logs.append(f"compose error: {e}")
            print(f"[agent] compose error: {e}")

    def _step_stop(self, state: AgentState) -> None:
        self._on_step("stop", "● Nessuna risposta utile")
        state.logs.append("stop: explicit no-action")
        state.aborted = True

    # ─────── Heuristics ───────
    @staticmethod
    def _guess_memory_relevance(query: str) -> bool:
        """Quick check senza chiamare LLM: il planner LLM è più sofisticato,
        ma quando il planner è in heuristic mode questo aiuta."""
        q = (query or "").lower()
        signals = (
            "come dicevo", "come avevo detto", "ricordi", "ne avevamo parlato",
            "l'altra volta", "ne abbiamo parlato", "come dicevamo",
            "as i mentioned", "as we discussed", "we said", "remember",
        )
        return any(s in q for s in signals)


# ──────────────────────────────────────────────────────────────
# Helper: query variants generator (mini-LLM call dedicato)
# ──────────────────────────────────────────────────────────────
_VARIANTS_SYSTEM_PROMPT = """Sei un generatore di varianti di query per retrieval RAG.
Ricevi una query (5-15 parole). Devi produrre 2-3 RIFORMULAZIONI DIVERSE che
preservino l'intento ma usino sinonimi/parafrasi/alternative lessicali.

REGOLE:
- Mantieni nomi propri (ProductA, ContactCenterSuite, Acme Consulting, ecc.) ESATTI.
- Cambia verbi/sostantivi con sinonimi. Es: "prezzo"→"costo", "tempo"→"durata".
- Cambia struttura sintattica. Es: domanda diretta → asserzione.
- Lingua: stessa dell'input.
- Output: SOLO le varianti, una per riga, niente numeri/bullet/preamboli.

ESEMPIO:
Input: prezzo versione base
Output:
costo versione standard
quanto costa il piano base
tariffa licenza entry-level

Input: integrazione ProductA con ContactCenterSuite
Output:
collegare ProductA a ContactCenterSuite
ProductA compatibile con ContactCenterSuite
connettività ProductA ContactCenterSuite
"""


def _generate_query_variants(query: str) -> list[str]:
    """Genera 2-3 varianti della query via mini-LLM call. ~300ms."""
    if not query or not query.strip():
        return []
    try:
        import ollama
        resp = ollama.chat(
            model=RAG.query_rewriting_model,
            messages=[
                {"role": "system", "content": _VARIANTS_SYSTEM_PROMPT},
                {"role": "user", "content": f"Input: {query}\nOutput:"},
            ],
            options={
                "temperature": 0.5,    # un po' di diversità
                "num_predict": 100,
            },
            keep_alive=LLM.ollama_keep_alive,
            stream=False,
        )
        raw = (resp.get("message", {}).get("content") or "").strip()
        if not raw:
            return []
        # Una variante per riga
        out = []
        for line in raw.split("\n"):
            line = line.strip().lstrip("-•*0123456789. ").strip('"\' ')
            if line and line.lower() != query.lower() and len(line) < 200:
                out.append(line)
            if len(out) >= 3:
                break
        return out
    except Exception as e:
        print(f"[variants] error: {e}")
        return []
