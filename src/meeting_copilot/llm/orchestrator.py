"""Orchestrator: gestisce il flusso Analyzer → Retriever → Composer.

Lavora in un thread separato. Espone:
- request_now(): triggera immediatamente il flusso (hotkey)
- set_auto_mode(bool): attiva/disattiva il polling periodico ogni N secondi
- stop(): segnale di chiusura

Comunica con la UI via callback (passati nel costruttore), che sotto il cofano
si traducono in pyqtSignal nell'overlay per essere thread-safe.
"""
from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

from ..config import LLM
from ..kb.manager import KnowledgeManager
from .analyzer import Analyzer, AnalysisResult
from .composer import Composer


@dataclass
class _PreparedCycle:
    """Risultato dello stage prep (analyzer + retrieval) — Latency #8.

    Passato dal prep_worker al compose_worker via queue.
    """
    analysis: AnalysisResult
    hits: list
    grounding_tier: str
    recent: list[str]
    analyzer_ms: float
    retr_ms: float
    t_start: float


class Orchestrator:
    def __init__(
        self,
        get_recent_lines: Callable[[], list[str]],
        on_status: Callable[[str, str], None],        # (status_text, color_hex)
        on_analysis: Callable[[AnalysisResult], None],
        on_token: Callable[[str], None],              # streaming delta
        on_complete: Callable[[str, list], None],     # full_text, hits
    ):
        """
        Args:
            get_recent_lines: chiamata dal thread orchestrator quando serve la
                              trascrizione recente. Deve essere thread-safe.
            on_status: callback per messaggi di stato ("Analyzing…", "Generating…").
            on_analysis: callback al termine dell'analyzer (anche se trigger=False).
            on_token: callback per ogni token streaming del composer.
            on_complete: callback alla fine, con testo completo + chunks RAG.
        """
        self._get_recent_lines = get_recent_lines
        self._on_status = on_status
        self._on_analysis = on_analysis
        self._on_token = on_token
        self._on_complete = on_complete

        self._analyzer = Analyzer()
        self._composer = Composer()

        self._stop = threading.Event()
        self._request_event = threading.Event()
        self._auto_mode = LLM.default_mode == "auto"
        self._lock = threading.Lock()

        # Latency #2: hint passed da consider_line() → _run_cycle()
        # Quando fast-path triggera con score alto, salva qui la query + categoria
        # detected per skippare la chiamata analyzer LLM (-1s).
        self._fast_path_hint: dict | None = None

        # Latency #8: pipelining state
        # _prep_q: prep results → compose_worker
        # max 2 perché non vogliamo accumulare prep stantii
        self._prep_q: "queue.Queue[_PreparedCycle]" = queue.Queue(maxsize=2)

        self._thread: Optional[threading.Thread] = None
        self._compose_thread: Optional[threading.Thread] = None

    # ─── API pubblica ───────────────────────────────────────
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        # Latency #8: usa pipelining se abilitato
        if getattr(LLM, "pipelining_enabled", False):
            self._thread = threading.Thread(target=self._run_loop_prep, daemon=True)
            self._compose_thread = threading.Thread(target=self._run_loop_compose, daemon=True)
            self._thread.start()
            self._compose_thread.start()
            print(f"[orch] started PIPELINED (prep+compose threads, auto={self._auto_mode})")
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            print(f"[orch] started (auto={self._auto_mode}, interval={LLM.auto_interval_s}s)")
        # Pre-warmup dei modelli in background: la prima call vera sarà veloce.
        threading.Thread(target=self._warmup, daemon=True).start()
        # Live rolling summary background (Context L3)
        if LLM.rolling_summary_interval_s > 0:
            threading.Thread(target=self._rolling_summary_loop, daemon=True).start()

    def _rolling_summary_loop(self) -> None:
        """Background loop: ogni N sec rigenera rolling summary della call.

        Reset summary all'avvio (nuova call = nuovo riassunto).
        """
        from ..meeting_summary import update_summary, clear_summary
        clear_summary()
        # Aspetta prima esecuzione: dai tempo alla call di accumulare contesto
        first_wait = max(LLM.rolling_summary_interval_s // 2, 60)
        time.sleep(first_wait)
        while not self._stop.is_set():
            try:
                recent = self._get_recent_lines()
                if len(recent) >= LLM.rolling_summary_min_lines:
                    update_summary(recent)
            except Exception as e:
                print(f"[orch] rolling_summary error: {e}")
            # Sleep fino al prossimo ciclo (interruttibile da stop)
            for _ in range(LLM.rolling_summary_interval_s):
                if self._stop.is_set():
                    return
                time.sleep(1)

    def _warmup(self) -> None:
        """Carica analyzer + composer in RAM con una mini-call dummy.
        keep_alive=-1 li mantiene su finché l'app vive.

        Skip per provider non-local (cloud/remote): cloud non ha warmup
        (model sempre disponibile), remote ha il suo warmup server-side.
        """
        from ..config import PROVIDERS
        if PROVIDERS.llm != "local":
            # Cloud / remote → no warmup necessario
            print(f"[orch] ✓ LLM provider={PROVIDERS.llm}, no warmup needed.")
            self._on_status("● LLM ready", "#10B981")
            return
        try:
            import ollama
            print(f"[orch] warming up {LLM.analyzer_model}…")
            ollama.chat(
                model=LLM.analyzer_model,
                messages=[{"role": "user", "content": "ok"}],
                options={"num_predict": 1},
                keep_alive=LLM.ollama_keep_alive,
                stream=False,
            )
            print(f"[orch] warming up {LLM.composer_model}…")
            ollama.chat(
                model=LLM.composer_model,
                messages=[{"role": "user", "content": "ok"}],
                options={"num_predict": 1},
                keep_alive=LLM.ollama_keep_alive,
                stream=False,
            )
            print("[orch] ✓ LLM models warm (kept in RAM).")
            self._on_status("● LLM ready", "#10B981")
        except Exception as e:
            print(f"[orch] warmup error: {e}")

    def stop(self) -> None:
        self._stop.set()
        self._request_event.set()
        # Latency #8: unblock compose_worker
        try:
            self._prep_q.put_nowait(None)  # type: ignore[arg-type]
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=3)
        if self._compose_thread:
            self._compose_thread.join(timeout=3)

    def request_now(self) -> None:
        """Triggera immediatamente il flusso (hotkey Cmd+Shift+R)."""
        self._request_event.set()

    def chat_request(self, text: str) -> None:
        """Chat mode: ciclo con testo utente come query (skip analyzer).

        Riusa la pipeline fast-path. Il prep_worker / _run_cycle vede
        l'hint, salta analyzer, fa retrieval con `text` come query,
        composer risponde con category=question + KB active.

        Args:
            text: domanda dell'utente nel chat input.
        """
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._fast_path_hint = {
                "query": text,
                "category": "question",
                "confidence": 1.0,
                "signal": "chat",
            }
        self._request_event.set()
        print(f"[orch] 💬 chat_request: {text[:80]!r}")

    # ─── Fast-path trigger pattern-based ────────────────────
    def consider_line(self, text: str) -> None:
        """Chiamato dall'overlay ad ogni nuova riga trascritta.

        Se il testo contiene un pattern signal forte (domanda esplicita,
        richiesta "spiegami", "?"), lancia SUBITO un ciclo bypassando il
        wait di auto_interval_s. Risparmia 1-2s di latency.

        Idempotent: rispetta cooldown per evitare spam.
        """
        if not LLM.fast_path_enabled:
            return
        if not text or not text.strip():
            return

        # Cooldown check
        now = time.monotonic()
        if hasattr(self, "_fast_path_last") and (now - self._fast_path_last) < LLM.fast_path_cooldown_s:
            return

        # Pattern signal check (Italian/English question + info-request)
        try:
            from .signals import detect_question, detect_info_request
            q = detect_question(text)
            ir = detect_info_request(text)
            top_score = max(q.score, ir.score)
            top_name = q.name if q.score >= ir.score else ir.name
        except Exception:
            return

        if top_score < LLM.fast_path_min_score:
            return

        print(f"[orch] ⚡ fast-path trigger ({top_name}, score={top_score:.2f}): {text[:60]!r}")
        self._fast_path_last = now
        # Latency #2: salva hint per skippare analyzer LLM in _run_cycle
        with self._lock:
            self._fast_path_hint = {
                "query": text.strip(),
                "category": "question",   # tutti i nostri pattern sono question-like
                "confidence": top_score,
                "signal": top_name,
            }
        self._request_event.set()

    def set_auto_mode(self, enabled: bool) -> None:
        with self._lock:
            self._auto_mode = enabled

    def get_auto_mode(self) -> bool:
        with self._lock:
            return self._auto_mode

    # ─── Latency #8: Pipelined loops (prep + compose threads) ───────
    def _run_loop_prep(self) -> None:
        """Thread A: runs analyzer + retrieval, pushes _PreparedCycle in _prep_q.

        Mentre il compose_worker streamea il ciclo N, qui partiamo già col
        prep del ciclo N+1 → -30-50% perceived latency.
        """
        while not self._stop.is_set():
            with self._lock:
                auto = self._auto_mode
            timeout = LLM.auto_interval_s if auto else None
            self._request_event.wait(timeout=timeout)
            self._request_event.clear()
            if self._stop.is_set():
                break
            try:
                prepared = self._do_prep()
            except Exception as e:
                print(f"[orch-prep] error: {e}")
                self._on_status(f"● Prep error: {e}", "#EF4444")
                continue
            if prepared is None:
                continue
            try:
                self._prep_q.put(prepared, timeout=1)
            except queue.Full:
                # Stale prep accumulato → drop il più vecchio e prova di nuovo
                try:
                    self._prep_q.get_nowait()
                    self._prep_q.put_nowait(prepared)
                except Exception:
                    pass

    def _run_loop_compose(self) -> None:
        """Thread B: drains _prep_q, esegue composer streaming sequenzialmente.

        Il compose è serializzato (un solo suggerimento mostrato per volta).
        """
        while not self._stop.is_set():
            try:
                prepared = self._prep_q.get(timeout=0.3)
            except queue.Empty:
                continue
            if prepared is None:
                break  # poison pill
            try:
                self._do_compose(prepared)
            except Exception as e:
                print(f"[orch-compose] error: {e}")
                self._on_status(f"● Composer error: {e}", "#EF4444")

    def _do_prep(self) -> Optional[_PreparedCycle]:
        """Stage 1+2: analyzer + retrieval. Ritorna _PreparedCycle o None se skip."""
        recent = self._get_recent_lines()
        print(f"[orch-prep] cycle: {len(recent)} recent lines")
        if not recent:
            return None
        active_sections = KnowledgeManager.get_active_sections()
        t0 = time.monotonic()

        # ─── STAGE 1: Analyzer (skippato su fast-path hint) ───
        with self._lock:
            hint = self._fast_path_hint
            self._fast_path_hint = None
        if hint is not None:
            analysis = AnalysisResult(
                trigger=True,
                category=hint["category"],
                query=hint["query"],
                confidence=hint["confidence"],
                raw=f"fast-path:{hint['signal']}",
            )
            analyzer_ms = (time.monotonic() - t0) * 1000
            print(
                f"[orch-prep] analyzer SKIPPED (fast-path {hint['signal']}, "
                f"{analyzer_ms:.0f}ms saved)"
            )
        else:
            self._on_status("● Analyzing…", "#F59E0B")
            try:
                analysis = self._analyzer.analyze(recent, active_sections)
            except Exception as e:
                print(f"[orch-prep] analyzer error: {e}")
                return None
            analyzer_ms = (time.monotonic() - t0) * 1000
            print(
                f"[orch-prep] analyzer: cat={analysis.category} conf={analysis.confidence:.2f} "
                f"trigger={analysis.trigger} ({analyzer_ms:.0f}ms)"
            )
        self._on_analysis(analysis)
        if not analysis.trigger:
            self._on_status(
                f"● {analysis.category} (conf {analysis.confidence:.2f})",
                "#6B7280",
            )
            return None

        # ─── STAGE 2: Retrieval ───
        self._on_status(f"● {analysis.category} → cerca contesto…", "#3B82F6")
        t_retr = time.monotonic()
        try:
            from ..config import RAG as _RAG
            _final_k = _RAG.final_top_k
            if _RAG.crag_enabled:
                from ..kb.retriever import search_with_correction
                hits, metadata, verdict = search_with_correction(
                    analysis.query, sections=active_sections, top_k=_final_k
                )
                grounding_tier = metadata.tier if _RAG.confidence_gating_enabled else "medium"
            else:
                hits = KnowledgeManager.search(
                    analysis.query, sections=active_sections, top_k=_final_k
                )
                if _RAG.confidence_gating_enabled:
                    from ..kb.retriever import _compute_grounding
                    grounding_tier = _compute_grounding(hits).tier
                else:
                    grounding_tier = "medium"
        except Exception as e:
            print(f"[orch-prep] retrieval error: {e}")
            return None
        retr_ms = (time.monotonic() - t_retr) * 1000
        print(f"[orch-prep] retrieval: {len(hits)} hits, tier={grounding_tier} ({retr_ms:.0f}ms)")

        return _PreparedCycle(
            analysis=analysis,
            hits=hits,
            grounding_tier=grounding_tier,
            recent=recent,
            analyzer_ms=analyzer_ms,
            retr_ms=retr_ms,
            t_start=t0,
        )

    def _do_compose(self, prepared: _PreparedCycle) -> None:
        """Stage 3: Composer streaming. Bloccante."""
        tier_emoji = {"high": "📎", "medium": "📎+🌐", "low": "🌐"}.get(prepared.grounding_tier, "")
        self._on_status(
            f"● Generating ({prepared.analysis.category}) {tier_emoji}…", "#10B981"
        )
        t1 = time.monotonic()
        _ttft_holder: list[float] = []
        _orig_on_token = self._on_token

        def _on_token_timed(piece: str) -> None:
            if not _ttft_holder:
                _ttft_holder.append(time.monotonic() - t1)
            if _orig_on_token:
                _orig_on_token(piece)

        try:
            full_text = self._composer.compose_stream(
                category=prepared.analysis.category,
                query=prepared.analysis.query,
                chunks=prepared.hits,
                transcript_lines=prepared.recent,
                on_token=_on_token_timed,
                grounding_tier=prepared.grounding_tier,
            )
        except Exception as e:
            print(f"[orch-compose] error: {e}")
            self._on_status(f"● Composer error: {e}", "#EF4444")
            return
        composer_s = time.monotonic() - t1
        ttft_ms = (_ttft_holder[0] * 1000) if _ttft_holder else composer_s * 1000
        total_s = time.monotonic() - prepared.t_start
        print(
            f"[orch-compose] TTFT={ttft_ms:.0f}ms full={composer_s:.2f}s "
            f"chars={len(full_text)} | TOTAL cycle={total_s:.2f}s "
            f"(pipelined: prep+compose may overlap)"
        )
        self._on_complete(full_text, prepared.hits)
        self._on_status(
            f"● Done ({prepared.analysis.category}, {composer_s:.1f}s)", "#10B981"
        )

    # ─── Loop principale (legacy, non-pipelined) ───────────────
    def _run_loop(self) -> None:
        while not self._stop.is_set():
            # Wait fino a: timeout auto-mode O hotkey
            with self._lock:
                auto = self._auto_mode
            timeout = LLM.auto_interval_s if auto else None

            triggered = self._request_event.wait(timeout=timeout)
            self._request_event.clear()

            if self._stop.is_set():
                break

            # Se è scaduto il timeout senza trigger manuale e siamo in auto:
            # esegui il ciclo periodico.
            # Se è arrivato un request_now (manuale): esegui comunque.
            self._run_cycle()

    def _run_cycle(self) -> None:
        recent = self._get_recent_lines()
        print(f"[orch] cycle: {len(recent)} recent lines")
        if not recent:
            return

        active_sections = KnowledgeManager.get_active_sections()
        print(f"[orch] active sections: {active_sections}")

        # ─── STAGE 1: Analyzer (skippato su fast-path hint) ───
        t0 = time.monotonic()
        with self._lock:
            hint = self._fast_path_hint
            self._fast_path_hint = None   # consume

        if hint is not None:
            # Latency #2: skip analyzer cloud (-1s). Pattern detector ha già
            # classificato → costruiamo direttamente l'AnalysisResult.
            from .analyzer import AnalysisResult
            analysis = AnalysisResult(
                trigger=True,
                category=hint["category"],
                query=hint["query"],
                confidence=hint["confidence"],
                raw=f"fast-path:{hint['signal']}",
            )
            analyzer_ms = (time.monotonic() - t0) * 1000
            print(
                f"[orch] analyzer SKIPPED (fast-path hint={hint['signal']}, "
                f"score={hint['confidence']:.2f}) query={hint['query'][:60]!r} "
                f"({analyzer_ms:.0f}ms saved)"
            )
        else:
            self._on_status("● Analyzing…", "#F59E0B")
            try:
                analysis = self._analyzer.analyze(recent, active_sections)
            except Exception as e:
                print(f"[orch] analyzer error: {e}")
                self._on_status(f"● Analyzer error: {e}", "#EF4444")
                return
            analyzer_ms = (time.monotonic() - t0) * 1000
            print(
                f"[orch] analyzer: category={analysis.category} "
                f"conf={analysis.confidence:.2f} trigger={analysis.trigger} "
                f"query={analysis.query[:60]!r} ({analyzer_ms:.0f}ms)"
            )

        self._on_analysis(analysis)

        if not analysis.trigger:
            # Niente di interessante: torna idle.
            self._on_status(
                f"● {analysis.category} (conf {analysis.confidence:.2f}, {analyzer_ms:.0f}ms)",
                "#6B7280",
            )
            return

        # ─── STAGE 2: Retrieval (Fase 14: con CRAG opzionale) ───
        self._on_status(
            f"● {analysis.category} → cerca contesto…", "#3B82F6"
        )
        t_retr = time.monotonic()
        try:
            from ..config import RAG as _RAG
            # top_k da config (default 3 dopo Latency#7), passato esplicito
            # per consistency con search_with_correction signature.
            _final_k = _RAG.final_top_k
            if _RAG.crag_enabled:
                # CRAG: retrieve → evaluate → re-query se serve
                self._on_status(f"● Valutando rilevanza…", "#3B82F6")
                from ..kb.retriever import search_with_correction
                hits, metadata, verdict = search_with_correction(
                    analysis.query, sections=active_sections, top_k=_final_k
                )
                grounding_tier = metadata.tier if _RAG.confidence_gating_enabled else "medium"
                print(f"[orch] CRAG verdict={verdict} tier={grounding_tier}")
            else:
                hits = KnowledgeManager.search(
                    analysis.query, sections=active_sections, top_k=_final_k
                )
                # Fase 13 baseline
                if _RAG.confidence_gating_enabled:
                    from ..kb.retriever import _compute_grounding
                    grounding_tier = _compute_grounding(hits).tier
                else:
                    grounding_tier = "medium"
        except Exception as e:
            print(f"[orch] retrieval error: {e}")
            self._on_status(f"● Retrieval error: {e}", "#EF4444")
            return
        retr_ms = (time.monotonic() - t_retr) * 1000
        print(
            f"[orch] retrieval: {len(hits)} hits, tier={grounding_tier} ({retr_ms:.0f}ms)"
        )

        # ─── STAGE 3: Composer (streaming) ───
        tier_emoji = {"high": "📎", "medium": "📎+🌐", "low": "🌐"}.get(grounding_tier, "")
        self._on_status(
            f"● Generating ({analysis.category}) {tier_emoji}…", "#10B981"
        )
        t1 = time.monotonic()
        # Wrap on_token per misurare TTFT (time-to-first-token)
        _ttft_holder: list[float] = []
        _orig_on_token = self._on_token

        def _on_token_timed(piece: str) -> None:
            if not _ttft_holder:
                _ttft_holder.append(time.monotonic() - t1)
            if _orig_on_token:
                _orig_on_token(piece)
        try:
            full_text = self._composer.compose_stream(
                category=analysis.category,
                query=analysis.query,
                chunks=hits,
                transcript_lines=recent,
                on_token=_on_token_timed,
                grounding_tier=grounding_tier,  # Fase 13
            )
        except Exception as e:
            print(f"[orch] composer error: {e}")
            self._on_status(f"● Composer error: {e}", "#EF4444")
            return
        composer_s = time.monotonic() - t1
        ttft_ms = (_ttft_holder[0] * 1000) if _ttft_holder else composer_s * 1000
        total_s = time.monotonic() - t0
        print(
            f"[orch] composer: TTFT={ttft_ms:.0f}ms, full={composer_s:.2f}s, "
            f"chars={len(full_text)} | TOTAL cycle={total_s:.2f}s"
        )

        self._on_complete(full_text, hits)
        self._on_status(
            f"● Done ({analysis.category}, {composer_s:.1f}s)", "#10B981"
        )
