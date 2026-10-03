"""Gestione paths della knowledge base."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from ..config import PROJECT_ROOT


# Root della KB
KB_ROOT = PROJECT_ROOT / "kb"
KB_ROOT.mkdir(parents=True, exist_ok=True)

# Stato attivo
ACTIVE_KB_FILE = Path.home() / ".config" / "meeting_copilot" / "active_kb.json"
ACTIVE_KB_FILE.parent.mkdir(parents=True, exist_ok=True)


def slugify(name: str) -> str:
    """Trasforma un nome umano (es. 'Pitch ProductB - Voicebot') in uno slug
    compatibile col filesystem ('pitch_productb_voicebot')."""
    # Normalizza unicode (rimuove accenti)
    s = unicodedata.normalize("NFKD", name)
    s = s.encode("ascii", "ignore").decode("ascii")
    # Lowercase, sostituisce non-alfanumerici con _, collassa _ multipli
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "section"


def section_dir(slug: str) -> Path:
    return KB_ROOT / slug


def sources_dir(slug: str) -> Path:
    return section_dir(slug) / "sources"


def chunks_file(slug: str) -> Path:
    return section_dir(slug) / "chunks.jsonl"


def index_dir(slug: str) -> Path:
    return section_dir(slug) / "index"


def metadata_file(slug: str) -> Path:
    return section_dir(slug) / "section.json"


def ensure_section_dirs(slug: str) -> None:
    """Crea la struttura base della sezione."""
    sources_dir(slug).mkdir(parents=True, exist_ok=True)
    index_dir(slug).mkdir(parents=True, exist_ok=True)


def list_section_slugs() -> list[str]:
    """Elenca tutti gli slug delle sezioni presenti su disco."""
    if not KB_ROOT.exists():
        return []
    return sorted(
        d.name for d in KB_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
