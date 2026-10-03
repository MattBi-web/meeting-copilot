"""Brain — piramide di sintesi (v1.1a, DESIGN_PRODUCT_V1_1.md §4).

Livelli:
    L0  transcript/chunk        → Chroma/BM25 (esiste, drill-down)
    L1  sintesi per meeting     → brain.db `meetings` (extractor.py)
    L2  digest per stream       → brain.db `stream_digests` (consolidate.py)
    L3  index card globale      → brain.db `meta` (consolidate.py)

Entry points:
    ingest_session(...)  → chiamata da ws_call a fine call: L1 + dirty flag
    python -m meeting_copilot.brain.consolidate  → cron notturno L2+L3
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger("meeting_copilot.brain")


def ingest_session(*, user_id: str, stream: str | None, session_id: str,
                   transcript_entries: list, duration_s: float,
                   language: str | None = "it") -> tuple[dict | None, str | None]:
    """Estrae L1 dal transcript e lo persiste. Ritorna (l1_dict, testo_reso).

    Best-effort: su qualsiasi errore ritorna (None, None) — mai bloccare
    la chiusura della call.
    """
    if not transcript_entries:
        return None, None
    try:
        from .extractor import extract_l1, render_l1_text
        from .store import BrainStore
        l1 = extract_l1(transcript_entries, language=language)
        if l1 is None:
            return None, None
        text = render_l1_text(l1)
        store = BrainStore.instance()
        meeting_id = store.add_meeting(
            user_id=user_id,
            stream=stream or "_unassigned",
            session_id=session_id,
            ts=time.time(),
            duration_s=int(duration_s),
            l1=l1,
        )
        # v1.1c hot path: meeting senza stream → classificazione automatica
        if not stream:
            try:
                from .taxonomy import classify_meeting
                res = classify_meeting(l1, meeting_id, user_id, store=store)
                log.info("taxonomy hot path: %s", res)
            except Exception:
                log.exception("taxonomy hot path fallita (best-effort)")
        return l1, text
    except Exception:
        log.exception("brain.ingest_session fallita (best-effort)")
        return None, None
