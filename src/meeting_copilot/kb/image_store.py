"""Image vector store + retrieval — Fase 20.3.

SQLite-based: tabella `images` con (id, scope, source_file, slide_idx, image_path,
embedding BLOB). Cosine search in Python.

API:
    rebuild_images(slug)       → riestrai PPTX + embedda + indicizza
    image_search(query, scopes) → top-K SlideHit per query
    images_stats(slug)         → conteggi

Pipeline rebuild:
    1. list pptx files in sources/
    2. for each: extract slides as PNG (parsers_images)
    3. embed batch via image_embedding provider
    4. upsert in SQLite store

Per-section database in section_dir(slug)/images.db
"""
from __future__ import annotations

import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import IMAGE
from . import paths


# ──────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SlideHit:
    """Risultato di image_search."""
    scope: str
    source_file: str
    slide_idx: int
    image_path: str
    score: float


# ──────────────────────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────────────────────
def _db_path(slug: str) -> Path:
    return paths.section_dir(slug) / "images.db"


def _emb_to_blob(emb: list[float]) -> bytes:
    if not emb:
        return b""
    return struct.pack(f"{len(emb)}f", *emb)


def _blob_to_emb(blob: bytes) -> list[float]:
    if not blob:
        return []
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL,
            source_file TEXT NOT NULL,
            slide_idx INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            embedding BLOB,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_images_scope ON images (scope);
        CREATE INDEX IF NOT EXISTS idx_images_source ON images (scope, source_file);
    """)


# ──────────────────────────────────────────────────────────────
# Rebuild — full pipeline
# ──────────────────────────────────────────────────────────────
def rebuild_images(slug: str, *, verbose: bool = True) -> dict:
    """Rebuild totale dell'index immagini per la sezione.

    1. Reset store
    2. Walk sources/ per .pptx
    3. Extract slides → PNG
    4. Embed batch
    5. Insert

    Returns:
        dict con stats (n_pptx, n_slides, elapsed_s).
    """
    if not IMAGE.enabled:
        return {"skipped": True, "reason": "IMAGE.enabled=False"}

    t0 = time.monotonic()
    src_dir = paths.sources_dir(slug)
    if not src_dir.exists():
        return {"error": f"sources dir non esiste: {src_dir}"}

    # Reset DB
    db_p = _db_path(slug)
    if db_p.exists():
        db_p.unlink()
    conn = sqlite3.connect(str(db_p))
    _init_schema(conn)

    # Find pptx
    pptx_files = list(src_dir.glob("*.pptx"))
    if verbose:
        print(f"[images] '{slug}': {len(pptx_files)} pptx trovati")

    if not pptx_files:
        conn.close()
        return {"n_pptx": 0, "n_slides": 0, "elapsed_s": time.monotonic() - t0}

    # Extract + embed
    from .parsers_images import extract_pptx_images
    from ..providers.local.image_embedding import LocalImageEmbeddingProvider

    embedder = LocalImageEmbeddingProvider()
    n_slides_total = 0

    for pptx_path in pptx_files:
        if verbose:
            print(f"[images]   → extract {pptx_path.name}")
        try:
            slides = extract_pptx_images(pptx_path, slug, prefer_render=True)
        except Exception as e:
            print(f"[images]     ✗ extraction failed: {e}")
            continue

        if not slides:
            continue

        if verbose:
            print(f"[images]   → embedding {len(slides)} slides…")
        try:
            embeddings = embedder.embed_images_batch(
                [s.image_path for s in slides]
            )
        except Exception as e:
            print(f"[images]     ✗ embedding failed: {e}")
            continue

        for s, emb in zip(slides, embeddings):
            if not emb:
                continue
            conn.execute(
                "INSERT INTO images (scope, source_file, slide_idx, image_path, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (slug, s.source_file, s.slide_idx, str(s.image_path),
                 _emb_to_blob(emb), time.time()),
            )
            n_slides_total += 1

        conn.commit()

    conn.close()
    elapsed = time.monotonic() - t0
    if verbose:
        print(f"[images] ✓ '{slug}': {len(pptx_files)} pptx, {n_slides_total} slides "
              f"({elapsed:.1f}s)")
    return {
        "n_pptx": len(pptx_files),
        "n_slides": n_slides_total,
        "elapsed_s": elapsed,
    }


# ──────────────────────────────────────────────────────────────
# Search
# ──────────────────────────────────────────────────────────────
def image_search(query: str, scopes: list[str], top_k: int | None = None) -> list[SlideHit]:
    """Cerca le slide più rilevanti per la query in tutti gli scope.

    Args:
        query: stringa testuale
        scopes: lista di slug sezioni
        top_k: default IMAGE.top_k

    Returns:
        list[SlideHit] ordinati per score desc, filtrati per min_similarity.
    """
    if not IMAGE.enabled or not query.strip() or not scopes:
        return []

    top_k = top_k if top_k is not None else IMAGE.top_k

    # Embed della query in spazio multimodale
    try:
        from ..providers.local.image_embedding import LocalImageEmbeddingProvider
        embedder = LocalImageEmbeddingProvider()
        q_emb = embedder.embed_text(query.strip())
    except Exception as e:
        print(f"[image_search] embed error: {e}")
        return []
    if not q_emb:
        return []

    all_hits: list[SlideHit] = []
    for scope in scopes:
        db_p = _db_path(scope)
        if not db_p.exists():
            continue
        with sqlite3.connect(str(db_p)) as conn:
            rows = conn.execute(
                "SELECT source_file, slide_idx, image_path, embedding FROM images "
                "WHERE scope = ?",
                (scope,),
            ).fetchall()
        for src_file, sidx, img_path, blob in rows:
            emb = _blob_to_emb(blob)
            score = _cosine(q_emb, emb)
            if score >= IMAGE.min_similarity:
                all_hits.append(SlideHit(
                    scope=scope, source_file=src_file, slide_idx=sidx,
                    image_path=img_path, score=score,
                ))

    all_hits.sort(key=lambda h: h.score, reverse=True)
    return all_hits[:top_k]


def images_stats(slug: str) -> dict:
    db_p = _db_path(slug)
    if not db_p.exists():
        return {"n_slides": 0, "indexed": False}
    with sqlite3.connect(str(db_p)) as conn:
        n = conn.execute("SELECT COUNT(*) FROM images WHERE scope = ?", (slug,)).fetchone()[0]
        files = conn.execute(
            "SELECT COUNT(DISTINCT source_file) FROM images WHERE scope = ?",
            (slug,),
        ).fetchone()[0]
    return {"n_slides": n, "n_pptx_files": files, "indexed": True}
