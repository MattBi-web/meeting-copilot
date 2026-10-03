"""Session manager: traccia tutto quello che succede in una call.

Una Session contiene:
- metadata: id, start, end, sezioni KB attive all'inizio/durante
- transcript_lines: tutte le righe trascritte con timestamp + lingua
- suggestions: ogni volta che il composer ha generato un suggerimento (con categoria, query, fonti)
- notes: appunti scritti dall'utente nel pannello notes
- saved_passages: passaggi salvati on-the-fly come chunks KB
- decisions: (futuro) decisioni esplicite marcate dall'utente

Persistente: il file `sessions/session_<timestamp>.json` viene salvato
incrementally dopo ogni nuova riga (in modo che un crash non perda dati).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .config import PROJECT_ROOT


SESSIONS_DIR = PROJECT_ROOT / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────
@dataclass
class TranscriptLine:
    timestamp: str           # HH:MM:SS
    text: str
    language: str = ""
    duration_s: float = 0.0


@dataclass
class SuggestionEntry:
    timestamp: str
    category: str            # question | objection | pitch_opportunity | noise
    query: str
    text: str                # output del composer
    sources: list[str] = field(default_factory=list)   # ["section/source.pdf", ...]
    confidence: float = 0.0


@dataclass
class SavedPassage:
    timestamp: str
    section_slug: str        # in quale sezione KB è stato salvato
    text: str


# ─────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────
class Session:
    """Single call session. Append-only, persist su disco dopo ogni mutazione."""

    def __init__(self, path: Path | None = None):
        self.id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.started_at = datetime.now().isoformat()
        self.ended_at: str | None = None
        # Path verrà rifinito quando set_initial_sections viene chiamato
        # (organizza per sezione attiva). Default: flat folder.
        self.path = path or (SESSIONS_DIR / f"session_{self.id}.json")
        self.initial_active_sections: list[str] = []
        self.transcript_lines: list[TranscriptLine] = []
        self.suggestions: list[SuggestionEntry] = []
        self.notes: str = ""
        self.saved_passages: list[SavedPassage] = []
        # Path al summary markdown se generato
        self.summary_path: str | None = None

    # ─── Mutators ───
    def set_initial_sections(self, sections: list[str]) -> None:
        self.initial_active_sections = list(sections)
        # Riorganizza path della sessione in cartella per-section,
        # così "sessions/<section>/session_<id>.json" invece di flat.
        # Se ci sono più sezioni attive, usa la prima per il bucket.
        if sections and not self._path_was_custom():
            primary_section = sections[0]
            section_dir = SESSIONS_DIR / primary_section
            section_dir.mkdir(parents=True, exist_ok=True)
            new_path = section_dir / f"session_{self.id}.json"
            # Migra il file se già scritto
            if self.path.exists() and self.path != new_path:
                try:
                    import shutil
                    shutil.move(str(self.path), str(new_path))
                except Exception as e:
                    print(f"[session] errore migrazione path: {e}")
            self.path = new_path
        self._save()

    def _path_was_custom(self) -> bool:
        """True se path è stato impostato esplicitamente dal costruttore."""
        # Heuristica: se è dentro SESSIONS_DIR/sessions e nome inizia con session_,
        # è il default → possiamo riorganizzare.
        try:
            self.path.relative_to(SESSIONS_DIR)
            return False
        except ValueError:
            return True

    def add_transcript_line(self, text: str, language: str = "", duration_s: float = 0.0) -> None:
        line = TranscriptLine(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            text=text,
            language=language,
            duration_s=duration_s,
        )
        self.transcript_lines.append(line)
        self._save()

    def add_suggestion(
        self,
        category: str,
        query: str,
        text: str,
        sources: list[str],
        confidence: float = 0.0,
    ) -> None:
        s = SuggestionEntry(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            category=category,
            query=query,
            text=text,
            sources=sources,
            confidence=confidence,
        )
        self.suggestions.append(s)
        self._save()

    def update_notes(self, text: str) -> None:
        self.notes = text
        self._save()

    def add_saved_passage(self, section_slug: str, text: str) -> None:
        p = SavedPassage(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            section_slug=section_slug,
            text=text,
        )
        self.saved_passages.append(p)
        self._save()

    def close(self) -> None:
        self.ended_at = datetime.now().isoformat()
        self._save()

    def set_summary_path(self, path: Path) -> None:
        self.summary_path = str(path)
        self._save()

    # ─── Serialization ───
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "initial_active_sections": self.initial_active_sections,
            "transcript_lines": [asdict(t) for t in self.transcript_lines],
            "suggestions": [asdict(s) for s in self.suggestions],
            "notes": self.notes,
            "saved_passages": [asdict(p) for p in self.saved_passages],
            "summary_path": self.summary_path,
        }

    def _save(self) -> None:
        # Atomic-ish write: prima su .tmp, poi rename. Evita corruzione su crash.
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

    # ─── Helpers ───
    def full_transcript_text(self) -> str:
        return "\n".join(
            f"[{t.timestamp}] {t.text}" for t in self.transcript_lines
        )

    def all_sources_cited(self) -> list[str]:
        seen: list[str] = []
        for s in self.suggestions:
            for src in s.sources:
                if src not in seen:
                    seen.append(src)
        return seen
