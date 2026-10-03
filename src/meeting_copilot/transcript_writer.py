"""Append-only writer per la trascrizione su file."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .config import TRANSCRIPTS_DIR


class TranscriptWriter:
    """Scrive ogni riga di trascrizione su un file timestamped, append-only.

    Il file viene aperto al primo write() e flushato dopo ogni riga, in modo
    che anche un crash o un Ctrl+C non perda l'ultimo passaggio.
    """

    def __init__(self, prefix: str = "transcript", out_dir: Path = TRANSCRIPTS_DIR):
        self.out_dir = out_dir
        self.out_dir.mkdir(exist_ok=True, parents=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.out_dir / f"{prefix}_{ts}.txt"
        self._fh = None

    def _open(self):
        if self._fh is None:
            self._fh = self.path.open("w", encoding="utf-8")
            self._fh.write(f"# Meeting Copilot transcript\n")
            self._fh.write(f"# Started: {datetime.now().isoformat()}\n\n")
            self._fh.flush()
        return self._fh

    def write(self, text: str, language: str = "", duration_s: float = 0.0) -> None:
        if not text:
            return
        fh = self._open()
        ts = datetime.now().strftime("%H:%M:%S")
        lang = f" [{language}]" if language else ""
        dur = f" ({duration_s:.1f}s)" if duration_s else ""
        fh.write(f"[{ts}]{lang}{dur} {text}\n")
        fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.write(f"\n# Ended: {datetime.now().isoformat()}\n")
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
