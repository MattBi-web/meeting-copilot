"""Session prompt store — load/save di meta-istruzioni iniettate nei composer.

Persistenza: PROJECT_ROOT/session_prompt.txt (semplice text file).
Lettura cached in memory + invalidazione su scrittura (no race conditions
gravi visto il low write rate).

API:
    get_session_prompt() → str (vuota se non set)
    set_session_prompt(text) → persiste
    clear_session_prompt() → rimuove

Helper:
    prepend_to_system(system_prompt) → str   (con separatore chiaro)
"""
from __future__ import annotations

import threading
from pathlib import Path

from .config import PROJECT_ROOT, SESSION


_lock = threading.Lock()
_cache: str | None = None


def _persist_path() -> Path:
    return PROJECT_ROOT / SESSION.persist_file


def get_session_prompt() -> str:
    """Ritorna il prompt corrente (vuoto se non set)."""
    global _cache
    if _cache is not None:
        return _cache
    p = _persist_path()
    if not p.exists():
        _cache = ""
        return ""
    try:
        _cache = p.read_text(encoding="utf-8").strip()
    except Exception:
        _cache = ""
    return _cache


def set_session_prompt(text: str) -> str:
    """Salva il prompt. Ritorna il testo applicato (potenzialmente troncato)."""
    global _cache
    text = (text or "").strip()
    if len(text) > SESSION.max_chars:
        text = text[: SESSION.max_chars]
    p = _persist_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        p.write_text(text, encoding="utf-8")
        _cache = text
    return text


def clear_session_prompt() -> None:
    global _cache
    p = _persist_path()
    with _lock:
        if p.exists():
            p.unlink()
        _cache = ""


def prepend_to_system(system_prompt: str) -> str:
    """Prependa il session prompt al system prompt del composer, se presente.

    Output: SESSION_CONTEXT block + separator + system_prompt originale.
    Quando session_prompt è vuoto, ritorna system_prompt invariato.
    """
    sp = get_session_prompt()
    if not sp:
        return system_prompt
    return (
        "=== SESSION CONTEXT (istruzioni meta-call dall'utente) ===\n"
        f"{sp}\n"
        "=== END SESSION CONTEXT ===\n\n"
        f"{system_prompt}"
    )
