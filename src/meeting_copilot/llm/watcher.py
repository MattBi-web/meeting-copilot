"""Watcher Agent: il cervello di "quando suggerire cosa".

Architettura ibrida (vedi piano Fase 6):
- L0 → pattern signals (regex/keywords), 1-5ms
- L1 → KB similarity (embedding + ChromaDB), 100-200ms
- Suppression → cooldown topic, frequency limit, self-speech, smalltalk
- L2 → LLM analyzer (llama3.2:3b) chiamato SOLO se score grezzo > soglia base
- L3 → LLM composer (qwen2.5:7b) chiamato in SPECULATIVE quando score > 0.55,
       o direttamente quando score > 0.75

Tier UI:
- score < 0.30  → silent (header grigio)
- 0.30 – 0.55   → indicator giallo (visibile ma niente UI)
- 0.55 – 0.75   → HOOK (gancio cliccabile) + speculative composer in background
- > 0.75        → AUTO-TRIGGER pannello suggerimenti completo

Speculative execution: ogni hook ha un id. Il composer parte subito in background.
Se l'utente clicca il gancio → mostra risultato cached (o aspetta). Se score sale
a > 0.75 → mostra automaticamente. Se passa N secondi senza utilizzo → discard.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from ..config import LLM, TELEMETRY, WATCHER
from ..kb.manager import KnowledgeManager
from ..kb.retriever import SearchHit
from .analyzer import Analyzer, AnalysisResult
from .composer import Composer
from .signals import L0Result, compute_l0
from .suppression import SuppressionManager


# ──────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────
@dataclass
class WatcherDecision:
    """Risultato di un'analisi del watcher su una nuova riga."""
    id: str                           # uuid univoco
    score: float                      # 0..1 finale
    tier: str                         # "silent" | "indicator" | "hook" | "auto"
    speaker: str                      # self|other|both|silence|unknown
    l0: L0Result
    l1_top_hit: Optional[SearchHit]   # best chunk se l'embedding match c'è
    l1_score: float
    suppression_penalty: float
    suppression_label: Optional[str]
    hook_label: str                   # breve testo del gancio (se tier=hook|auto)
    query: str                        # query estratta per RAG/composer
    chunks: list[SearchHit] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    # Fase 13: grounding tier per source-aware composer
    grounding_tier: str = "medium"    # "high" | "medium" | "low"


@dataclass
class SpeculativeResult:
    decision_id: str
    text: str
    hits: list[SearchHit]
    done: bool


# ──────────────────────────────────────────────────────────────
# Watcher Agent
# ──────────────────────────────────────────────────────────────
class WatcherAgent:
    def __init__(
        self,
        on_decision: Callable[[WatcherDecision], None],
        on_token: Callable[[str, str], None],          # (decision_id, piece)
        on_complete: Callable[[str, str, list], None], # (decision_id, full_text, hits)
        on_status: Callable[[str, str], None],         # (text, color_hex)
        sections: list[str] | None = None,             # M3: KB scopes espliciti
    ):
        self.on_decision = on_decision
        self.on_token = on_token
        self.on_complete = on_complete
        self.on_status = on_status
        # M3 multi-tenant: se sections è fornito, il watcher usa SOLO quelle
        # (già prefissate per-user via tenant.user_scope). Se None, fallback
        # allo stato globale KnowledgeManager (comportamento overlay V0).
        self.sections = sections

        self._suppression = SuppressionManager()
        self._analyzer = Analyzer()
        self._composer = Composer()

        # Speculative cache: decision_id → SpeculativeResult
        self._speculative: dict[str, SpeculativeResult] = {}
        # Lock per accesso concorrente
        self._lock = threading.Lock()
        # Pool di worker speculativi
        self._speculative_threads: dict[str, threading.Thread] = {}

        self._enabled = True

        # Fase 10: contatore per ricomputo periodico adaptive thresholds
        self._n_decisions_since_recompute = 0

    def _active_sections(self) -> list[str]:
        """M3: sezioni esplicite dell'istanza, o fallback global (V0)."""
        if self.sections is not None:
            return self.sections
        return KnowledgeManager.get_active_sections()

    # ── API pubblica ──
    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = enabled

    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def consider(self, recent_lines: list[str], speaker_label: str) -> WatcherDecision:
        """Valuta le ultime righe. Chiamare ogni volta che arriva una nuova riga
        trascritta. Non blocca: per gli stage costosi usa thread.
        """
        if not self.is_enabled():
            return self._silent_decision(speaker_label, "watcher disabled")

        # 1. Self-suppression
        if speaker_label == "self":
            self._suppression.note_self_speech()
        if self._suppression.is_in_self_grace():
            return self._silent_decision(speaker_label, "self_speech_grace")

        # 2. L0 signals
        l0 = compute_l0(recent_lines)

        # 3. L1 KB similarity
        l1_score, l1_top_hit, all_hits, query_embed = _compute_l1(
            recent_lines, sections=self._active_sections())

        # Fase 13 — derive grounding tier from hits (before composer call)
        grounding_tier = _derive_grounding_tier(all_hits)

        # 4. Suppression: topic cooldown
        cooldown_penalty, cooldown_label = self._suppression.topic_cooldown_penalty(query_embed)

        # 5. Aggregato
        # Score grezzo = w0 * salience_L0 + w1 * similarity_L1
        salience = max(l0.salience_score, 0.0)
        # boost: se entrambi sono attivi insieme, l'effetto è amplificato (AND-like)
        combined = WATCHER.w0_pattern * salience + WATCHER.w1_kb * l1_score
        if salience > 0.4 and l1_score > 0.4:
            combined += 0.1  # both-active bonus

        # Sottrai suppressioni
        combined -= l0.suppression_score * 0.6
        combined -= cooldown_penalty * 0.5

        # Rate limit: se >max trigger/min, forza max tier "hook"
        rate_limited = self._suppression.is_rate_limited()

        # Clamp finale
        final_score = max(0.0, min(1.0, combined))

        # Tier mapping con soglie ADAPTIVE (Fase 10) che possono variare a runtime
        thresholds = _get_runtime_thresholds()
        if rate_limited and final_score > thresholds.threshold_hook:
            tier = "hook"   # forza hook anche se sarebbe auto
            reasons = ["rate_limited"]
        elif final_score < thresholds.threshold_indicator:
            tier = "silent"
            reasons = []
        elif final_score < thresholds.threshold_hook:
            tier = "indicator"
            reasons = []
        elif final_score < thresholds.threshold_auto:
            tier = "hook"
            reasons = []
        else:
            tier = "auto"
            reasons = []

        # Query per RAG/composer
        query = _extract_query(l0, recent_lines)
        hook_label = _build_hook_label(l0, query, l1_top_hit)

        decision_id = uuid.uuid4().hex[:8]
        decision = WatcherDecision(
            id=decision_id,
            score=final_score,
            tier=tier,
            speaker=speaker_label,
            l0=l0,
            l1_top_hit=l1_top_hit,
            l1_score=l1_score,
            suppression_penalty=cooldown_penalty,
            suppression_label=cooldown_label,
            hook_label=hook_label,
            query=query,
            chunks=all_hits,
            reasons=reasons,
            grounding_tier=grounding_tier,
        )

        # Telemetry: record SEMPRE (anche silent serve per analytics)
        try:
            from ..telemetry import record_decision
            record_decision(decision)
        except Exception:
            pass

        # Notifica sempre (UI può aggiornare indicator)
        self.on_decision(decision)

        # Speculative + auto
        if tier in ("hook", "auto"):
            self._launch_speculative(decision)
        if tier == "auto":
            self._deliver_speculative_now(decision)

        # Ricomputo periodico adaptive thresholds (Fase 10)
        if tier != "silent":
            self._n_decisions_since_recompute += 1
            if self._n_decisions_since_recompute >= TELEMETRY.recompute_every_n:
                self._n_decisions_since_recompute = 0
                try:
                    from ..telemetry import recompute_thresholds
                    recompute_thresholds(reason="periodic")
                except Exception:
                    pass

        return decision

    # ── Speculative ──
    def _launch_speculative(self, decision: WatcherDecision) -> None:
        """Avvia in background L2 (analyzer per ratifica) e L3 (composer streaming)."""
        with self._lock:
            self._speculative[decision.id] = SpeculativeResult(
                decision_id=decision.id, text="", hits=decision.chunks, done=False
            )
        t = threading.Thread(
            target=self._run_speculative, args=(decision,), daemon=True
        )
        with self._lock:
            self._speculative_threads[decision.id] = t
        t.start()

    def _run_speculative(self, decision: WatcherDecision) -> None:
        try:
            # L2 ratifica (opzionale, può essere skipped per velocità)
            if WATCHER.l2_ratify:
                analysis = self._analyzer.analyze(
                    [decision.query] if decision.query else [],
                    self._active_sections(),
                )
                if not analysis.trigger and decision.tier == "hook":
                    # L2 dice "noise" → annulla speculative
                    self._discard_speculative(decision.id, reason="L2 noise")
                    return
                category = analysis.category if analysis.trigger else "question"
            else:
                # Senza L2: assume categoria from L0 signals
                category = _guess_category(decision.l0)

            # ─── Branch: Agentic vs Legacy compose ───
            from ..config import RAG as _RAG
            if _RAG.agentic_mode_enabled:
                self._run_agentic(decision, category)
            else:
                self._run_legacy(decision, category)

            # Aggiorna cooldown per il topic appena risposto
            _, _, _, embed = _compute_l1(
                [decision.query], sections=self._active_sections())
            if embed:
                self._suppression.add_cooldown_topic(embed, decision.hook_label[:80])
            self._suppression.note_full_trigger()
        except Exception as e:
            self.on_status(f"● Speculative error: {e}", "#EF4444")

    # ─── Legacy compose path (Fase 6-15) ───
    def _run_legacy(self, decision: WatcherDecision, category: str) -> None:
        def on_piece(piece: str) -> None:
            with self._lock:
                spec = self._speculative.get(decision.id)
                if spec is None or spec.done:
                    return
                spec.text += piece
            self.on_token(decision.id, piece)

        # Fase 15: aggiungi memorie rilevanti da scope = sezioni KB attive
        memory_context = _fetch_memory_context(
            decision.query, self._active_sections()
        )
        tx_extra = [memory_context] if memory_context else []

        full_text = self._composer.compose_stream(
            category=category,
            query=decision.query,
            chunks=decision.chunks,
            transcript_lines=tx_extra,
            on_token=on_piece,
            grounding_tier=decision.grounding_tier,
        )
        with self._lock:
            spec = self._speculative.get(decision.id)
            if spec is not None:
                spec.text = full_text
                spec.done = True

        self.on_complete(decision.id, full_text, decision.chunks)

    # ─── Fase 16: Agentic compose path ───
    def _run_agentic(self, decision: WatcherDecision, category: str) -> None:
        """Usa AgentOrchestrator: Planner decide gli step, esegue stepwise.

        Sostituisce il singolo compose_stream con un flusso multi-step.
        Lo streaming token va sempre alla stessa UI callback.
        """
        from .agent_orchestrator import AgentOrchestrator

        def on_piece(piece: str) -> None:
            with self._lock:
                spec = self._speculative.get(decision.id)
                if spec is None or spec.done:
                    return
                spec.text += piece
            self.on_token(decision.id, piece)

        def on_step(step_name: str, status_text: str) -> None:
            # Aggiorna lo status header dell'UI con il passo corrente
            self.on_status(status_text, "#3B82F6")
            print(f"[agent] step → {step_name}: {status_text}")

        agent = AgentOrchestrator(
            on_token=on_piece,
            on_step=on_step,
            composer=self._composer,
        )

        result = agent.run(
            query=decision.query,
            sections=self._active_sections(),
            category=category,
            transcript_lines=[],   # le memorie sono fetchate dentro retrieve_memory step
        )

        with self._lock:
            spec = self._speculative.get(decision.id)
            if spec is not None:
                spec.text = result.response_text
                spec.hits = result.hits
                spec.done = True

        # Mostra brevemente il plan eseguito nel log overlay
        plan_summary = " → ".join(result.plan.steps)
        print(f"[agent] decision {decision.id}: plan=[{plan_summary}] "
              f"elapsed={result.elapsed_ms:.0f}ms success={result.success}")

        self.on_complete(decision.id, result.response_text, result.hits)

    def _deliver_speculative_now(self, decision: WatcherDecision) -> None:
        """Per tier='auto': l'UI riceve subito il segnale che deve mostrare
        questo decision_id non appena i token arrivano. La UI lo gestisce.
        Qui semplicemente loggiamo per debug."""
        print(f"[watcher] AUTO-FIRE decision {decision.id} score={decision.score:.2f}")

    def _discard_speculative(self, decision_id: str, reason: str = "") -> None:
        with self._lock:
            self._speculative.pop(decision_id, None)
            self._speculative_threads.pop(decision_id, None)

    # ── User actions ──
    def user_clicked_hook(self, decision_id: str) -> SpeculativeResult | None:
        """Quando l'utente clicca un gancio, ritorna la versione cached
        (anche se ancora in progress). UI gestirà l'aspetta-completamento."""
        with self._lock:
            return self._speculative.get(decision_id)

    # ── Helpers ──
    def _silent_decision(self, speaker: str, reason: str) -> WatcherDecision:
        d = WatcherDecision(
            id=uuid.uuid4().hex[:8],
            score=0.0,
            tier="silent",
            speaker=speaker,
            l0=L0Result(signals={}, salience_score=0.0, suppression_score=1.0, notes=[reason]),
            l1_top_hit=None,
            l1_score=0.0,
            suppression_penalty=0.0,
            suppression_label=None,
            hook_label="",
            query="",
        )
        self.on_decision(d)
        return d


# ──────────────────────────────────────────────────────────────
# L1 — KB similarity gate (no LLM, solo embedding + ChromaDB)
# Fase 14: ora opzionalmente usa Corrective RAG (CRAG)
# ──────────────────────────────────────────────────────────────
def _compute_l1(
    recent_lines: list[str],
    sections: list[str] | None = None,
) -> tuple[float, Optional[SearchHit], list[SearchHit], list[float]]:
    """Ritorna (top_score, top_hit, all_top_k, query_embedding).

    M3: `sections` esplicite (scopes per-user). None = fallback global (V0).
    Quando RAG.crag_enabled è True, usa search_with_correction (self-corrective).
    Altrimenti usa search standard.
    """
    if not recent_lines:
        return (0.0, None, [], [])

    if sections is None:
        sections = KnowledgeManager.get_active_sections()
    if not sections:
        return (0.0, None, [], [])

    # Query = ultime 3 righe (~ ultimi 30s di conversazione)
    query = " ".join(recent_lines[-3:]).strip()
    if not query:
        return (0.0, None, [], [])

    try:
        from ..config import RAG as _RAG
        if _RAG.crag_enabled:
            # Fase 14: pipeline con self-correction
            from ..kb.retriever import search_with_correction
            hits, _metadata, _verdict = search_with_correction(
                query, sections=sections, top_k=3
            )
        else:
            hits = KnowledgeManager.search(query, sections=sections, top_k=3)
    except Exception as e:
        print(f"[watcher] _compute_l1 retrieval error: {e}")
        return (0.0, None, [], [])

    if not hits:
        return (0.0, None, [], [])

    # Recupera anche l'embedding della query per il cooldown (chiama Ollama)
    try:
        from ..kb.retriever import _embed_query
        embed = _embed_query(query)
    except Exception:
        embed = []

    top = hits[0]
    # BUGFIX 2026-07-05: con hybrid RRF top.score è un rank-score (~0.016),
    # che azzerava il contributo L1 (w1_kb * 0.016 ≈ 0). Il segnale di
    # similarity vero è il cosine dense preservato in metadata["dense_score"].
    l1_effective = float((top.metadata or {}).get("dense_score", top.score))
    return (l1_effective, top, hits, embed)


# ──────────────────────────────────────────────────────────────
# Helpers vari
# ──────────────────────────────────────────────────────────────
def _extract_query(l0: L0Result, recent_lines: list[str]) -> str:
    """Estrae la frase chiave per il RAG retrieval e il composer."""
    if not recent_lines:
        return ""
    # Strategia semplice: ultima frase con punto interrogativo, altrimenti l'ultima.
    last = recent_lines[-1].strip()
    for line in reversed(recent_lines[-3:]):
        if "?" in line:
            return line.strip()
    return last


def _build_hook_label(l0: L0Result, query: str, top_hit: SearchHit | None) -> str:
    """Costruisce una label breve per il "gancio" cliccabile (max 80 char)."""
    # Categoria dal L0 + indicazione del topic
    if l0.signals.get("question", None) and l0.signals["question"].score >= 0.5:
        prefix = "❓ Domanda"
    elif l0.signals.get("objection", None) and l0.signals["objection"].score >= 0.5:
        prefix = "⚠️ Obiezione"
    elif l0.signals.get("info_request", None) and l0.signals["info_request"].score >= 0.5:
        prefix = "📝 Info"
    elif l0.signals.get("named_entity", None) and l0.signals["named_entity"].score >= 0.5:
        prefix = "🎯 Spunto"
    else:
        prefix = "💡"

    snippet = query[:50] + ("…" if len(query) > 50 else "")
    if top_hit:
        return f"{prefix} · {snippet}  →  {top_hit.section}"
    return f"{prefix} · {snippet}"


def _fetch_memory_context(query: str, scopes: list[str]) -> str:
    """Fase 15: fetch memorie rilevanti come context extra al composer."""
    try:
        from ..config import MEMORY
        if not MEMORY.enabled or not scopes:
            return ""
        from ..memory import MemoryManager
        memories = MemoryManager.search(
            query, scopes, top_k=MEMORY.max_memories_per_query
        )
        if not memories:
            return ""
        context = MemoryManager.format_as_context(memories)
        print(f"[memory] retrieved {len(memories)} memorie per la query")
        return context
    except Exception as e:
        print(f"[memory] retrieval error: {e}")
        return ""


def _derive_grounding_tier(hits: list[SearchHit]) -> str:
    """Fase 13: usa kb.retriever._compute_grounding per derivare tier."""
    try:
        from ..kb.retriever import _compute_grounding
        from ..config import RAG as _RAG
        if not _RAG.confidence_gating_enabled:
            return "medium"  # behavior pre-Fase-13
        meta = _compute_grounding(hits)
        return meta.tier
    except Exception:
        return "medium"


def _get_runtime_thresholds():
    """Fase 10: ritorna le soglie correnti (adaptive se telemetry attiva,
    altrimenti statiche dal WatcherConfig). Pattern fallback-safe."""
    try:
        from ..telemetry import get_adaptive_thresholds
        return get_adaptive_thresholds()
    except Exception:
        # Fallback ai default statici (oggetto duck-typed)
        class _Static:
            threshold_indicator = WATCHER.threshold_indicator
            threshold_hook = WATCHER.threshold_hook
            threshold_auto = WATCHER.threshold_auto
            rrf_dense_weight = 0.7
            rrf_sparse_weight = 0.3
        return _Static()


def _guess_category(l0: L0Result) -> str:
    """Mapping L0 → categoria composer (senza chiamare L2)."""
    sigs = l0.signals
    if sigs.get("question") and sigs["question"].score >= 0.5:
        return "question"
    if sigs.get("objection") and sigs["objection"].score >= 0.5:
        return "objection"
    return "pitch_opportunity"
