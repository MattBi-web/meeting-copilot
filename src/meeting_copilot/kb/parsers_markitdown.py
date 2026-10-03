"""Parser layout-aware via Microsoft MarkitDown (Fase 9.8).

MarkitDown unifica TUTTI i formati (PDF, DOCX, PPTX, XLSX, HTML, images con OCR,
audio con Whisper, ecc.) in markdown ben strutturato. Vantaggi vs i parser
specifici attuali:

- Heading preservata in markdown → meglio per il chunking semantic
- Tabelle in formato `| col1 | col2 |` invece di linearizzate
- OCR integrato per PDF scansionati
- Una sola dipendenza invece di 4 (pypdf + python-docx + python-pptx + openpyxl)

Trade-off:
- Più pesante (~150MB di dipendenze)
- Più lento (file-by-file vs streaming)
- Output non sempre identico ai parser specifici → richiede re-index

Strategy: provider opt-in. Se installato + config.RAG.parser_preference != "specific",
viene usato. Altrimenti fallback ai parser specifici.
"""
from __future__ import annotations

from pathlib import Path

from .parsers import RawDoc


def is_markitdown_available() -> bool:
    try:
        import markitdown  # noqa: F401
        return True
    except ImportError:
        return False


def parse_file_markitdown(path: Path) -> list[RawDoc]:
    """Usa MarkitDown per parsare un file in markdown unificato."""
    try:
        from markitdown import MarkItDown
    except ImportError:
        raise RuntimeError("markitdown non installato. uv pip install markitdown[all]")

    md = MarkItDown(enable_plugins=False)
    try:
        result = md.convert(str(path))
    except Exception as e:
        raise RuntimeError(f"markitdown convert error: {e}") from e

    text = (result.text_content or "").strip()
    if not text:
        return []

    ext = path.suffix.lower().lstrip(".") or "unknown"
    return [RawDoc(
        text=text,
        metadata={
            "source": path.name,
            "type": ext,
            "parser": "markitdown",
        },
    )]


# Estensioni supportate da MarkitDown
MARKITDOWN_EXTS = (
    ".pdf", ".docx", ".pptx", ".xlsx", ".xls",
    ".md", ".markdown", ".txt", ".html", ".htm",
    ".csv", ".json", ".xml",
    # MarkitDown OCR (richiede extras): .png .jpg .gif
)


def supports_markitdown(path: Path) -> bool:
    return path.suffix.lower() in MARKITDOWN_EXTS
