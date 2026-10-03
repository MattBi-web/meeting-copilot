"""Indicizzazione: file → chunks → embeddings → ChromaDB.

Pipeline:
1. Walk della cartella `sources/` della sezione
2. Per ogni file supportato → parsers.parse_file() → lista RawDoc
3. Per ogni RawDoc → chunking (overlap sliding window)
4. Per ogni chunk → embedding via Ollama (nomic-embed-text)
5. Scrittura su ChromaDB collection persistente
6. Salvataggio JSONL `chunks.jsonl` (per audit/rebuild)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


from ..config import RAG
from . import paths
from .parsers import (
    RawDoc,
    SUPPORTED_EXTS,
    parse_file,
    parse_url,
)
from .parsers_markitdown import (
    is_markitdown_available,
    parse_file_markitdown,
    supports_markitdown,
)


def _dispatch_parse_file(entry):
    """Dispatcher parser secondo config.RAG.parser_preference.
    Fallback automatico ai parser specifici se MarkitDown non installato."""
    pref = RAG.parser_preference  # "auto" | "markitdown" | "specific"
    if pref == "specific":
        return parse_file(entry)
    if pref == "markitdown":
        if not is_markitdown_available():
            raise RuntimeError("parser_preference='markitdown' ma libreria non installata")
        if supports_markitdown(entry):
            return parse_file_markitdown(entry)
        return parse_file(entry)
    # auto
    if is_markitdown_available() and supports_markitdown(entry):
        try:
            return parse_file_markitdown(entry)
        except Exception as e:
            print(f"[kb] markitdown fallita su {entry.name}: {e}, fallback parser specifico")
            return parse_file(entry)
    return parse_file(entry)


# Embedding model: ora letto da RAG.embedding_model.
# Manteniamo costante di backward-compat per moduli legacy che importano EMBEDDING_MODEL.
EMBEDDING_MODEL = RAG.embedding_model

# Chunking
CHUNK_SIZE_CHARS = 800       # ~150-200 token, sweet spot per nomic-embed-text
CHUNK_OVERLAP_CHARS = 120    # overlap per non spezzare frasi a metà


@dataclass
class Chunk:
    """Chunk indicizzato. Identificato da `id` univoco."""
    id: str
    text: str
    source: str             # nome file o URL
    source_type: str        # pdf, docx, pptx, xlsx, md, txt, url
    section: str            # slug sezione di appartenenza
    metadata: dict          # page, slide, sheet, ecc.


# ─────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────
def _chunk_text(text: str, size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Sliding window con overlap. Cerca di tagliare su limiti di frase/paragrafo
    dove possibile (heuristic: backtrack al newline o punto più vicino)."""
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + size, n)
        # Cerca un limite naturale (newline o punto) entro 80 char dalla fine
        if j < n:
            for sep in ("\n\n", "\n", ". ", "! ", "? "):
                k = text.rfind(sep, i + size - 100, j)
                if k > i:
                    j = k + len(sep)
                    break
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        i = j - overlap if j < n else j
        if i <= 0:
            i = j  # safety
    return chunks


def _chunk_doc(doc: RawDoc, section: str, doc_idx: int = 0) -> list[Chunk]:
    """Chunk un RawDoc. `doc_idx` è l'indice del documento nel batch
    di reindex — serve a garantire ID unici anche quando due chunks
    hanno testo identico (es. slide template ripetute in PPTX)."""
    out: list[Chunk] = []
    for chunk_idx, chunk_text in enumerate(_chunk_text(doc.text)):
        # ID stabile + univoco: include posizione (doc_idx, chunk_idx).
        # Anche due chunks con stesso testo avranno ID diverso.
        h = hashlib.sha1(
            f"{section}|{doc.metadata.get('source', '')}|"
            f"{doc_idx}|{chunk_idx}|{chunk_text}".encode("utf-8")
        ).hexdigest()[:16]
        out.append(Chunk(
            id=f"{section}_{h}",
            text=chunk_text,
            source=doc.metadata.get("source", "unknown"),
            source_type=doc.metadata.get("type", "unknown"),
            section=section,
            metadata={k: v for k, v in doc.metadata.items() if k not in ("source", "type")},
        ))
    return out


# ─────────────────────────────────────────────────────────────
# Embeddings via Provider (Fase 8 refactor)
# ─────────────────────────────────────────────────────────────
def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts via configured embedding provider (default: local Ollama)."""
    from ..providers import get_embedding_provider
    provider = get_embedding_provider()
    return provider.embed_batch(texts)


# ─────────────────────────────────────────────────────────────
# Backward compat: legacy callers use _get_collection direct on ChromaDB
# ─────────────────────────────────────────────────────────────
def _get_collection(section_slug: str):
    """[LEGACY] ChromaDB collection accessor. Nuovo codice usa
    providers.get_vector_store_provider() invece."""
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        raise RuntimeError("chromadb non installato. uv pip install -e .")

    client = chromadb.PersistentClient(
        path=str(paths.index_dir(section_slug)),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    return client.get_or_create_collection(
        name=section_slug,
        metadata={"hnsw:space": "cosine"},
    )


# ─────────────────────────────────────────────────────────────
# Build/rebuild dell'indice di una sezione
# ─────────────────────────────────────────────────────────────
def _sources_fingerprint(src_dir) -> str:
    """Fingerprint dei sources: nome+size+mtime per file, contenuto _urls.txt.

    v1.1c hash gate: se identico all'ultimo reindex → skip totale.
    """
    import hashlib
    h = hashlib.sha256()
    try:
        for entry in sorted(src_dir.iterdir()):
            if entry.is_dir() or entry.name.startswith("."):
                continue
            if entry.name == "_urls.txt":
                h.update(entry.read_bytes())
                continue
            st = entry.stat()
            h.update(f"{entry.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    except FileNotFoundError:
        pass
    return h.hexdigest()


def reindex_section(slug: str, *, verbose: bool = True) -> dict:
    """Rebuild completo dell'indice della sezione.

    Steps:
    1. Walk sources/, parsa ogni file supportato
    2. Legge _urls.txt se presente, fetch + parse
    3. Chunking + embeddings
    4. Sovrascrive ChromaDB collection e chunks.jsonl

    Ritorna un dict con conteggi (sources, raw_docs, chunks, elapsed_s).
    """
    paths.ensure_section_dirs(slug)
    src_dir = paths.sources_dir(slug)

    t0 = time.monotonic()

    # v1.1c — hash gate: se NESSUN source è cambiato dall'ultimo reindex,
    # skip totale (niente re-embedding dell'invariato). Il fingerprint copre
    # nomi+size+mtime dei file supportati e il contenuto di _urls.txt.
    # (Incrementale per-source: TODO v1.2 — richiede delete/insert per source
    # nella collection Chroma.)
    fingerprint = _sources_fingerprint(src_dir)
    fp_file = paths.section_dir(slug) / ".sources_fingerprint"
    try:
        if fp_file.exists() and fp_file.read_text() == fingerprint:
            if verbose:
                print(f"[kb] reindex '{slug}': sources invariati → skip (hash gate)")
            return {"slug": slug, "skipped": True, "reason": "unchanged",
                    "elapsed_s": time.monotonic() - t0}
    except Exception:
        pass  # fingerprint illeggibile → reindex normale

    if verbose:
        print(f"[kb] reindex '{slug}' (sources: {src_dir})")

    # 1+2. Parse di tutti i file e di _urls.txt
    raw_docs: list[RawDoc] = []
    n_sources = 0

    for entry in sorted(src_dir.iterdir()):
        if entry.is_dir() or entry.name.startswith("."):
            continue

        # _urls.txt: un URL per riga
        if entry.name == "_urls.txt":
            urls = [
                line.strip() for line in entry.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            for url in urls:
                if verbose:
                    print(f"  → fetch URL: {url}")
                try:
                    raw_docs.extend(parse_url(url))
                    n_sources += 1
                except Exception as e:
                    print(f"    ✗ errore: {e}", file=sys.stderr)
            continue

        if entry.suffix.lower() not in SUPPORTED_EXTS:
            if verbose:
                print(f"  – skip (ext non supportato): {entry.name}")
            continue

        if verbose:
            print(f"  → parse: {entry.name}")
        try:
            raw_docs.extend(_dispatch_parse_file(entry))
            n_sources += 1
        except Exception as e:
            print(f"    ✗ errore: {e}", file=sys.stderr)

    if verbose:
        print(f"[kb] {n_sources} sources → {len(raw_docs)} raw docs")

    # 3. Chunking — passa doc_idx per garantire ID univoci anche su chunks duplicati
    all_chunks: list[Chunk] = []
    for doc_idx, doc in enumerate(raw_docs):
        all_chunks.extend(_chunk_doc(doc, slug, doc_idx=doc_idx))

    # Safety net: dedup finale per ID (paranoid check)
    seen_ids: set[str] = set()
    unique_chunks: list[Chunk] = []
    for c in all_chunks:
        if c.id in seen_ids:
            continue
        seen_ids.add(c.id)
        unique_chunks.append(c)
    if len(unique_chunks) < len(all_chunks):
        dropped = len(all_chunks) - len(unique_chunks)
        print(f"[kb] {dropped} chunks duplicati droppati per ID collision (safety)")
    all_chunks = unique_chunks

    if verbose:
        print(f"[kb] {len(all_chunks)} chunks generati. Computing embeddings…")

    # 4. Embeddings
    if all_chunks:
        embeddings = _embed_texts([c.text for c in all_chunks])
    else:
        embeddings = []

    # 5. Scrittura su provider (dense + sparse). Reset prima.
    from ..providers import get_vector_store_provider
    from ..providers.factory import get_sparse_store_provider

    vec_store = get_vector_store_provider()
    sparse_store = get_sparse_store_provider()

    vec_store.delete_section(slug)
    sparse_store.delete_section(slug)

    metas_list = [
        {
            "source": c.source,
            "source_type": c.source_type,
            "section": c.section,
            **{k: str(v) for k, v in c.metadata.items()},
        }
        for c in all_chunks
    ]

    if all_chunks:
        ids_list = [c.id for c in all_chunks]
        docs_list = [c.text for c in all_chunks]

        # Dense
        vec_store.upsert(
            section=slug,
            ids=ids_list,
            embeddings=embeddings,
            documents=docs_list,
            metadatas=metas_list,
        )
        # Sparse BM25 (Fase 9.6) — opt-in via config
        if RAG.hybrid_retrieval_enabled:
            try:
                sparse_store.upsert(
                    section=slug,
                    ids=ids_list,
                    documents=docs_list,
                    metadatas=metas_list,
                )
            except Exception as e:
                print(f"[kb] BM25 upsert skipped: {e}")

    # 6. Salva chunks.jsonl (snapshot human-readable)
    chunks_path = paths.chunks_file(slug)
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    # 7. Salva metadata sezione
    meta = {
        "slug": slug,
        "n_sources": n_sources,
        "n_chunks": len(all_chunks),
        "last_reindex": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": time.monotonic() - t0,
    }
    paths.metadata_file(slug).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 8. v1.1c — persisti il fingerprint per l'hash gate del prossimo reindex
    try:
        fp_file.write_text(fingerprint, encoding="utf-8")
    except Exception:
        pass

    if verbose:
        print(
            f"[kb] ✓ reindex '{slug}' fatto: "
            f"{n_sources} sources, {len(all_chunks)} chunks, "
            f"{meta['elapsed_s']:.1f}s"
        )

    return meta
