"""API alto-livello per la knowledge base.

Punto unico di accesso per CLI, UI e overlay. Mantiene anche lo stato
"sezioni attive" (quali sono attualmente abilitate per il retrieval).
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .indexer import Chunk, reindex_section
from .retriever import SearchHit, search


@dataclass
class Section:
    slug: str
    n_sources: int
    n_chunks: int
    last_reindex: str | None
    indexed: bool             # True se chunks > 0
    sources_dir: Path


# ─────────────────────────────────────────────────────────────
# Active sections state
# ─────────────────────────────────────────────────────────────
def get_active_sections() -> list[str]:
    """Quali sezioni sono attualmente attivate?"""
    if not paths.ACTIVE_KB_FILE.exists():
        return []
    try:
        data = json.loads(paths.ACTIVE_KB_FILE.read_text(encoding="utf-8"))
        return list(data.get("active", []))
    except Exception:
        return []


def set_active_sections(slugs: list[str]) -> None:
    paths.ACTIVE_KB_FILE.write_text(
        json.dumps({"active": slugs}, indent=2),
        encoding="utf-8",
    )


def toggle_section_active(slug: str) -> bool:
    """Toggle. Ritorna nuovo stato (True = attivata)."""
    active = get_active_sections()
    if slug in active:
        active.remove(slug)
        new_state = False
    else:
        active.append(slug)
        new_state = True
    set_active_sections(active)
    return new_state


# ─────────────────────────────────────────────────────────────
# KnowledgeManager — facade
# ─────────────────────────────────────────────────────────────
class KnowledgeManager:
    """Facade. UI e CLI usano questa classe."""

    @staticmethod
    def list_sections() -> list[Section]:
        out: list[Section] = []
        for slug in paths.list_section_slugs():
            meta_file = paths.metadata_file(slug)
            meta = {}
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            n_chunks = int(meta.get("n_chunks", 0))
            out.append(Section(
                slug=slug,
                n_sources=int(meta.get("n_sources", 0)),
                n_chunks=n_chunks,
                last_reindex=meta.get("last_reindex"),
                indexed=n_chunks > 0,
                sources_dir=paths.sources_dir(slug),
            ))
        return out

    @staticmethod
    def create_section(name: str) -> Section:
        slug = paths.slugify(name)
        if not slug:
            raise ValueError(f"Nome sezione vuoto/non valido: {name!r}")
        paths.ensure_section_dirs(slug)
        # Crea _urls.txt placeholder con commento
        urls_file = paths.sources_dir(slug) / "_urls.txt"
        if not urls_file.exists():
            urls_file.write_text(
                "# Un URL per riga. Verranno scaricati e indicizzati al prossimo reindex.\n"
                "# Esempio:\n"
                "# https://example.com/blog/post\n",
                encoding="utf-8",
            )
        return Section(
            slug=slug,
            n_sources=0,
            n_chunks=0,
            last_reindex=None,
            indexed=False,
            sources_dir=paths.sources_dir(slug),
        )

    @staticmethod
    def add_files(slug: str, file_paths: list[Path]) -> list[Path]:
        """Copia file dentro sources/. Non fa reindex automatico."""
        paths.ensure_section_dirs(slug)
        dst_dir = paths.sources_dir(slug)
        copied: list[Path] = []
        for src in file_paths:
            if not src.exists() or not src.is_file():
                continue
            dst = dst_dir / src.name
            # Evita sovrascrivere silenziosamente: aggiungi suffix se collide
            if dst.exists():
                stem = dst.stem
                suffix = dst.suffix
                i = 1
                while True:
                    candidate = dst_dir / f"{stem}_{i}{suffix}"
                    if not candidate.exists():
                        dst = candidate
                        break
                    i += 1
            shutil.copy2(src, dst)
            copied.append(dst)
        return copied

    @staticmethod
    def reindex(slug: str, verbose: bool = True) -> dict:
        return reindex_section(slug, verbose=verbose)

    @staticmethod
    def delete_section(slug: str) -> None:
        """Elimina TUTTA la sezione (sources + index). Operazione distruttiva."""
        section = paths.section_dir(slug)
        if section.exists():
            shutil.rmtree(section)
        # Rimuovi dalle attive se presente
        active = get_active_sections()
        if slug in active:
            active.remove(slug)
            set_active_sections(active)

    @staticmethod
    def search(query: str, sections: list[str] | None = None, top_k: int = 3) -> list[SearchHit]:
        """Search su sezioni date (default = sezioni attualmente attive)."""
        if sections is None:
            sections = get_active_sections()
        if not sections:
            return []
        return search(query, sections, top_k=top_k)

    @staticmethod
    def get_active_sections() -> list[str]:
        return get_active_sections()

    @staticmethod
    def set_active_sections(slugs: list[str]) -> None:
        set_active_sections(slugs)

    @staticmethod
    def toggle_section_active(slug: str) -> bool:
        return toggle_section_active(slug)
