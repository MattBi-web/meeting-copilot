"""Graph Builder — Fase 17.3.

Orchestra la pipeline: chunks.jsonl → extract → merge entità simili → store.

Hook post-indexing: chiamato esplicitamente via CLI (`mc graph rebuild <slug>`)
o via flag in reindex_section (Fase 17.5). Il grafo è OPT-IN, di base la
reindex normale NON costruisce il grafo per evitare overhead.

Merge dedup:
- Le entità vengono inserite case-insensitive sul name (gestito da upsert_entity).
- Aggiuntivamente, si fa cosine merging: due entità con embedding cosine > threshold
  vengono fuse (la meno citata punta alla più citata).

Parallelismo: extraction è il bottleneck (~300ms/chunk). Si può usare un thread
pool, ma Ollama serializza internamente — quindi 2-4 worker max danno benefit.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...config import GRAPH
from .. import paths
from .extractor import ExtractionResult, extract_from_chunk
from .store import GraphStore, _cosine


def _embed_entity_name(name: str) -> list[float]:
    """Embed name via provider corrente. Restituisce [] in caso di errore."""
    try:
        from ...providers import get_embedding_provider
        provider = get_embedding_provider()
        embeddings = provider.embed_batch([name])
        return embeddings[0] if embeddings else []
    except Exception as e:
        print(f"[graph_builder] embed error '{name}': {e}")
        return []


def _load_chunks(slug: str) -> list[dict]:
    """Carica chunks.jsonl della sezione (snapshot human-readable)."""
    cp = paths.chunks_file(slug)
    if not cp.exists():
        return []
    out: list[dict] = []
    with cp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _post_merge_similar_entities(store: GraphStore) -> int:
    """Dopo l'inserimento, fa un pass di merge cosine-based per fondere
    entità con embedding > threshold (es. "voicebot" ~ "voice bot").

    Strategia:
      - Carica tutte le entità con embedding
      - Ordina per mention_count desc (canonica = la più citata)
      - Per ciascuna, cerca duplicati cosine > threshold con mention_count minore
      - Rimappa le relations / mentions della duplicate sulla canonica
      - Elimina la duplicate

    Ritorna numero di fusioni effettuate.
    """
    threshold = GRAPH.entity_merge_threshold
    entities = [e for e in store.list_entities() if e.embedding]
    if len(entities) < 2:
        return 0

    merged = 0
    # entities è già ordinato per mention_count desc grazie a list_entities()
    canonical: list = []
    canonical_ids: set[int] = set()
    for e in entities:
        # Cerca se c'è un canonico già selezionato con cosine > threshold
        match = None
        for c in canonical:
            if _cosine(c.embedding, e.embedding) >= threshold:
                match = c
                break
        if match is None:
            canonical.append(e)
            canonical_ids.add(e.id)
        else:
            # Merge e → match
            _merge_entity_into(store, src_id=e.id, dst_id=match.id)
            merged += 1
    return merged


def _merge_entity_into(store: GraphStore, src_id: int, dst_id: int) -> None:
    """Rimappa relations & mentions di src su dst, poi elimina src."""
    if src_id == dst_id:
        return
    with store._conn() as c:  # noqa: SLF001
        # Mentions: cambia entity_id, gestendo conflitti con INSERT OR IGNORE
        c.execute(
            "UPDATE OR IGNORE mentions SET entity_id = ? WHERE entity_id = ?",
            (dst_id, src_id),
        )
        c.execute("DELETE FROM mentions WHERE entity_id = ?", (src_id,))

        # Relations come src: rimappa
        c.execute(
            "UPDATE OR IGNORE relations SET src_id = ? WHERE src_id = ?",
            (dst_id, src_id),
        )
        c.execute("DELETE FROM relations WHERE src_id = ?", (src_id,))

        # Relations come dst: rimappa
        c.execute(
            "UPDATE OR IGNORE relations SET dst_id = ? WHERE dst_id = ?",
            (dst_id, src_id),
        )
        c.execute("DELETE FROM relations WHERE dst_id = ?", (src_id,))

        # Trasferisci mention_count
        c.execute(
            "UPDATE entities SET mention_count = mention_count + "
            "(SELECT mention_count FROM entities WHERE id = ?) "
            "WHERE id = ?",
            (src_id, dst_id),
        )

        # Cancella src
        c.execute("DELETE FROM entities WHERE id = ?", (src_id,))


def build_graph(slug: str, *, verbose: bool = True, max_workers: int = 3) -> dict:
    """Costruisce o ricostruisce il grafo per la sezione `slug`.

    Steps:
      1. Carica chunks dal jsonl
      2. Reset graph store (idempotente)
      3. Extraction in parallelo (thread pool, max_workers worker)
      4. Inserisce entità (con embedding) + relazioni + mentions
      5. Pass di merge cosine duplicates

    Returns:
        dict con stats finali (entities, relations, mentions, elapsed_s).
    """
    if not GRAPH.enabled:
        if verbose:
            print(f"[graph] GRAPH.enabled=False, skipping build for '{slug}'")
        return {"entities": 0, "relations": 0, "mentions": 0, "elapsed_s": 0.0, "skipped": True}

    t0 = time.monotonic()
    chunks = _load_chunks(slug)
    if not chunks:
        if verbose:
            print(f"[graph] '{slug}': no chunks found in jsonl, skip")
        return {"entities": 0, "relations": 0, "mentions": 0, "elapsed_s": 0.0}

    if verbose:
        print(f"[graph] building '{slug}': {len(chunks)} chunks → extracting…")

    store = GraphStore(slug)
    store.reset()

    # 3. Extraction parallela
    def _process(idx_chunk: tuple[int, dict]) -> tuple[dict, ExtractionResult]:
        i, chunk = idx_chunk
        result = extract_from_chunk(chunk.get("text", ""))
        if verbose and (i + 1) % 10 == 0:
            print(f"[graph]   ...{i+1}/{len(chunks)} chunks extracted")
        return chunk, result

    extracted: list[tuple[dict, ExtractionResult]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for r in ex.map(_process, enumerate(chunks)):
            extracted.append(r)

    # 4. Inserisce entità → relazioni → mentions
    if verbose:
        print(f"[graph] inserting entities + relations into SQLite…")

    # Mapping locale "entity name lowercase" → id per relazioni dello stesso chunk
    for chunk, ext in extracted:
        chunk_id = chunk.get("id", "")
        source = chunk.get("source", "")

        local_name_to_id: dict[str, int] = {}
        for ent in ext.entities:
            emb = _embed_entity_name(ent.name)
            eid = store.upsert_entity(ent.name, ent.type, embedding=emb)
            if eid > 0:
                local_name_to_id[ent.name.lower()] = eid
                store.add_mention(eid, chunk_id, source)

        for rel in ext.relations:
            sid = local_name_to_id.get(rel.src.lower(), 0)
            did = local_name_to_id.get(rel.dst.lower(), 0)
            if sid > 0 and did > 0:
                store.upsert_relation(sid, did, rel.label, evidence_chunk_id=chunk_id)

    # 5. Merge cosine duplicates
    if verbose:
        print(f"[graph] post-merge cosine duplicates…")
    merged = _post_merge_similar_entities(store)
    if verbose and merged:
        print(f"[graph]   merged {merged} duplicate entities")

    stats = store.stats()
    stats["elapsed_s"] = time.monotonic() - t0
    stats["chunks"] = len(chunks)
    stats["merged"] = merged

    if verbose:
        print(
            f"[graph] ✓ '{slug}' done: "
            f"{stats['entities']} entities, {stats['relations']} relations, "
            f"{stats['mentions']} mentions ({stats['elapsed_s']:.1f}s)"
        )

    return stats
