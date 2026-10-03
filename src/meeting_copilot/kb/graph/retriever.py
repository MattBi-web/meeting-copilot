"""GraphRAG retriever — Fase 17.4.

Data una query, recupera chunks dal grafo seguendo:
  1. Embed query
  2. Match top-K entità per cosine sim su entity embeddings
  3. Espande 1-hop (o `expansion_hops`) per ottenere i vicini
  4. Recupera chunk_ids che menzionano qualsiasi entità nel sottografo
  5. Carica i chunks da chunks.jsonl e costruisce VectorHit list

Output: list[VectorHit] con `score` basato su:
  - boost = somma cosine_sim delle entità matchate nel chunk
  - normalizzato 0..1

Le caller (kb.retriever.search) faranno fusione RRF con i risultati dense+sparse.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ...config import GRAPH
from ...providers import get_embedding_provider
from ...providers.base import VectorHit
from .. import paths
from .store import GraphStore


# ──────────────────────────────────────────────────────────────
# Chunk lookup helpers (mini-cache per sezione)
# ──────────────────────────────────────────────────────────────
def _load_chunks_index(slug: str) -> dict[str, dict]:
    """Mappa chunk_id → record completo (text, source, source_type, metadata)."""
    cp = paths.chunks_file(slug)
    if not cp.exists():
        return {}
    out: dict[str, dict] = {}
    with cp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cid = rec.get("id")
                if cid:
                    out[cid] = rec
            except Exception:
                continue
    return out


# Cache TTL leggera: rebuild graph invalida il jsonl, ma è raro.
_chunks_index_cache: dict[str, tuple[float, dict]] = {}


def _cached_chunks_index(slug: str) -> dict[str, dict]:
    cp = paths.chunks_file(slug)
    if not cp.exists():
        return {}
    mtime = cp.stat().st_mtime
    cached = _chunks_index_cache.get(slug)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    idx = _load_chunks_index(slug)
    _chunks_index_cache[slug] = (mtime, idx)
    return idx


def _embed_query(query: str) -> list[float]:
    try:
        provider = get_embedding_provider()
        embeds = provider.embed_batch([query])
        return embeds[0] if embeds else []
    except Exception as e:
        print(f"[graph_retriever] embed error: {e}")
        return []


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────
def graph_search(
    query: str,
    sections: list[str],
    top_k: int | None = None,
) -> list[VectorHit]:
    """Search via graph 1-hop expansion. Aggrega su tutti gli scope.

    Args:
        query: stringa query (idealmente già rewritten).
        sections: lista di scope (slug sezione KB).
        top_k: numero massimo di chunks da ritornare (default GRAPH.max_graph_chunks).

    Returns:
        list[VectorHit] ordinati per score desc.
    """
    if not GRAPH.enabled or not query.strip() or not sections:
        return []

    top_k = top_k if top_k is not None else GRAPH.max_graph_chunks

    query_embed = _embed_query(query.strip())
    if not query_embed:
        return []

    # Aggrega chunks da tutti gli scope
    aggregated: dict[str, tuple[VectorHit, float]] = {}  # chunk_id → (hit, boost)

    for scope in sections:
        store = GraphStore(scope)
        # Step 1: top entità per cosine match
        entity_matches = store.find_entities_by_embedding(query_embed, top_k=5)
        if not entity_matches:
            continue

        # Step 2: espande con neighbors
        seed_ids = [e.id for e, _ in entity_matches]
        seed_scores = {e.id: s for e, s in entity_matches}

        expanded_ids: set[int] = set(seed_ids)
        for sid in seed_ids:
            for nid in store.neighbors(sid, hops=GRAPH.expansion_hops):
                # Vicini ereditano metà dello score dell'entità seed
                if nid not in seed_scores:
                    seed_scores[nid] = seed_scores[sid] * 0.5
                expanded_ids.add(nid)

        # Step 3: chunks che menzionano qualsiasi entità del sottografo
        chunk_ids = store.chunks_for_entities(list(expanded_ids))
        if not chunk_ids:
            continue

        # Step 4: carica testo dei chunks da chunks.jsonl
        chunks_idx = _cached_chunks_index(scope)

        # Boost per chunk: somma score delle entità presenti nel chunk
        # (chunks che toccano più entità rilevanti pesano di più)
        for cid in chunk_ids:
            rec = chunks_idx.get(cid)
            if not rec:
                continue

            # Quante entità seed/vicine sono in questo chunk? Costoso fare lookup
            # nelle mentions: lo facciamo solo top-N chunks. Per ora boost
            # uniforme = max score tra le entità che lo citano.
            # Ottimizzazione future: JOIN su mentions per query unica.
            with store._conn() as c:  # noqa: SLF001
                rows = c.execute(
                    "SELECT entity_id FROM mentions "
                    "WHERE scope = ? AND chunk_id = ?",
                    (scope, cid),
                ).fetchall()
            ent_ids_in_chunk = [r[0] for r in rows]
            if not ent_ids_in_chunk:
                continue
            boost = sum(seed_scores.get(eid, 0.0) for eid in ent_ids_in_chunk
                        if eid in expanded_ids)
            # Normalizzazione soft (max 1.0)
            score = min(1.0, boost)

            hit = VectorHit(
                text=rec.get("text", ""),
                source=rec.get("source", ""),
                source_type=rec.get("source_type", "unknown"),
                section=scope,
                score=score,
                metadata={**rec.get("metadata", {}), "_graph_retrieved": True},
            )
            existing = aggregated.get(cid)
            if existing is None or score > existing[1]:
                aggregated[cid] = (hit, score)

    # Sort + cap
    sorted_hits = sorted(aggregated.values(), key=lambda t: t[1], reverse=True)
    return [h for h, _s in sorted_hits[:top_k]]


def graph_stats(scope: str) -> dict:
    """Wrapper conveniente per il CLI."""
    return GraphStore(scope).stats()
