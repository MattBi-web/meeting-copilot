"""Live rolling summary del meeting — Context L3.

Mantiene un riassunto rolling della call corrente. Aggiornato ogni N minuti
da un mini LLM call (Claude Haiku 4.5, ~$0.001/call). Iniettato nel composer
come "background" prima di ogni risposta — così il composer "ricorda" tutta
la call anche se dura 1h+.

Storage: in memoria (singleton) + persisted su PROJECT_ROOT/meeting_summary.txt
per debug + visibilità dashboard.

API:
    get_summary() → str (vuoto all'inizio della call)
    update_summary(transcript_lines: list[str]) → str  (chiamato dall'orchestrator)
    clear_summary() → reset
"""
from __future__ import annotations

import threading
from pathlib import Path

from .config import PROJECT_ROOT


_lock = threading.Lock()
_cache: str = ""
_PERSIST_FILE = PROJECT_ROOT / "meeting_summary.txt"


_SUMMARY_SYSTEM_PROMPT = """Sei un AI che riassume in real-time conversazioni di call B2B.
Ricevi il transcript completo della call fin qui (con label [TU] e [ALTRO] per
i partecipanti). Devi produrre un riassunto di 3-6 bullet point, max 150 parole,
che catturi:

1. **Topic principali** discussi (es. "architettura multi-agent", "pricing", "GDPR")
2. **Decisioni / commitment** presi (es. "demo fissata per giovedì", "PoC entro Q3")
3. **Obiezioni / preoccupazioni** dell'interlocutore (es. "timeline 8 settimane vista lunga")
4. **Domande aperte** non ancora risolte
5. **Riferimenti tecnici** specifici (es. "menzione di ContactCenterSuite integration",
   "dato pricing $0.0043/min")

Output: SOLO i bullet, niente preamboli ("Ecco il riassunto..."). Lingua: stessa
del transcript (italiano se la call è in IT).

Se la call è agli inizi (pochi turni), produci 1-2 bullet brevi su quello che è stato
detto. Non inventare nulla."""


def get_summary() -> str:
    """Ritorna il summary corrente. Vuoto se non ancora generato."""
    global _cache
    if _cache:
        return _cache
    # Lazy-load da disco se cache vuota
    if _PERSIST_FILE.exists():
        try:
            _cache = _PERSIST_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            _cache = ""
    return _cache


def update_summary(transcript_lines: list[str]) -> str:
    """Rigenera il summary dal transcript corrente.

    Chiamato dal background loop dell'orchestrator (ogni 5 min) oppure
    manualmente. Usa il provider LLM corrente (cloud o local).

    Args:
        transcript_lines: lista di righe trascritte (già con speaker label).

    Returns:
        Il nuovo summary generato (può essere "" se errore).
    """
    global _cache
    if not transcript_lines:
        return _cache

    transcript_text = "\n".join(transcript_lines[-100:])  # cap a 100 ultimi turni
    if len(transcript_text) < 200:
        # Conversazione troppo corta, niente da riassumere
        return _cache

    try:
        from .providers import get_llm_provider
        from .providers.base import LLMMessage
        provider = get_llm_provider()
        user_msg = f"Transcript della call (più recente in basso):\n\n{transcript_text}\n\nRiassumi."
        result = provider.complete(
            model="llama3.2:3b",  # caller-side, viene rimappato a Haiku se cloud
            messages=[
                LLMMessage("system", _SUMMARY_SYSTEM_PROMPT),
                LLMMessage("user", user_msg),
            ],
            temperature=0.2,
            max_tokens=300,
        ).strip()
        if not result:
            return _cache
        with _lock:
            _cache = result
            try:
                _PERSIST_FILE.write_text(result, encoding="utf-8")
            except Exception:
                pass
        print(f"[summary] aggiornato ({len(result)} chars)")
        return result
    except Exception as e:
        print(f"[summary] error: {e}")
        return _cache


def clear_summary() -> None:
    """Reset summary a inizio call."""
    global _cache
    with _lock:
        _cache = ""
        try:
            if _PERSIST_FILE.exists():
                _PERSIST_FILE.unlink()
        except Exception:
            pass


def prepend_to_user_msg(user_msg: str) -> str:
    """Helper per composer: prepend del summary al user message se presente.

    Pattern di iniezione: il summary viene messo come "BACKGROUND" prima del
    contenuto della call recente. NON modifica il system prompt (che ha già
    le sue regole grounding/anti-hallucination).
    """
    s = get_summary()
    if not s:
        return user_msg
    return (
        "=== BACKGROUND CALL (rolling summary aggiornato) ===\n"
        f"{s}\n"
        "=== END BACKGROUND ===\n\n"
        f"{user_msg}"
    )
