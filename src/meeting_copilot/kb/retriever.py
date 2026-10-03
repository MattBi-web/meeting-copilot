"""Retrieval pipeline (Fase 7 + Fase 9.6).

Pipeline:
1. Query rewriting (LLM cleanup, opt-in)             [Fase 7.1]
2. Dense retrieval per sezione (vector store)        [Fase 3]
3. Sparse retrieval per sezione (BM25)               [Fase 9.6]
4. Hybrid fusion via Reciprocal Rank Fusion (RRF)    [Fase 9.6]
5. Score threshold filtering                         [Fase 7.2]
6. Deduplication                                     [Fase 7.2]
7. Cross-encoder rerank (opt-in)                     [Fase 7.3]
8. Final top-k

Tutti gli step opt-in via RAGConfig. Tutto via provider abstraction (Fase 8).
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Literal

from ..config import RAG
from ..providers import (
    get_embedding_provider,
    get_rerank_provider,
    get_vector_store_provider,
)
from ..providers.base import VectorHit
from ..providers.factory import get_sparse_store_provider


# Backward compat: ri-esportiamo SearchHit come alias di VectorHit per i moduli legacy
SearchHit = VectorHit


# ──────────────────────────────────────────────────────────────
# Latency #11 — Cache LRU embeddings query
# ──────────────────────────────────────────────────────────────
# Cache thread-safe per gli embedding di query. Hit rate atteso ~25-30%
# (utente ripete domande simili, LLM analyzer estrae stessa query rewritten).
# Salva ~450ms per hit (skip chiamata HTTP embed remote / Ollama local).
_EMBED_CACHE_MAX = 256
_embed_cache: "OrderedDict[str, list[float]]" = OrderedDict()
_embed_cache_lock = threading.Lock()
_embed_cache_hits = 0
_embed_cache_misses = 0


def _cached_embed(text: str) -> list[float]:
    """Embed con cache LRU thread-safe."""
    global _embed_cache_hits, _embed_cache_misses
    key = text.strip()
    if not key:
        return get_embedding_provider().embed(text)
    with _embed_cache_lock:
        cached = _embed_cache.get(key)
        if cached is not None:
            # LRU: move to end (most recently used)
            _embed_cache.move_to_end(key)
            _embed_cache_hits += 1
            if (_embed_cache_hits + _embed_cache_misses) % 10 == 0:
                hit_rate = _embed_cache_hits / max(1, _embed_cache_hits + _embed_cache_misses)
                print(f"[embed-cache] hits={_embed_cache_hits} miss={_embed_cache_misses} "
                      f"hit_rate={hit_rate:.1%}")
            return cached
    # Miss: compute outside lock (slow), then store
    emb = get_embedding_provider().embed(text)
    with _embed_cache_lock:
        _embed_cache[key] = emb
        if len(_embed_cache) > _EMBED_CACHE_MAX:
            _embed_cache.popitem(last=False)  # evict LRU
        _embed_cache_misses += 1
    return emb


# ──────────────────────────────────────────────────────────────
# Fase 13.1 — Grounding confidence metadata
# ──────────────────────────────────────────────────────────────
GroundingTier = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class GroundingMetadata:
    """Quality dell'evidence RAG. Permette al composer di decidere
    se rispondere grounded (KB), ibrido (KB + LLM general), o LLM-only."""

    tier: GroundingTier            # "high" | "medium" | "low"
    max_score: float               # score migliore tra i top hits (0..1)
    mean_score: float              # media degli score
    n_hits: int                    # totale chunks recuperati
    n_above_threshold: int         # chunks con score >= medium threshold
    explanation: str = ""          # debug/log

    @property
    def is_grounded(self) -> bool:
        """True se confidence è abbastanza alta per grounded answer."""
        return self.tier in ("high", "medium")

    @property
    def is_strong(self) -> bool:
        return self.tier == "high"


def _compute_grounding(hits: list[VectorHit]) -> GroundingMetadata:
    """Calcola grounding confidence dai top hits.

    Soglie configurabili in RAG.grounding_high_threshold / grounding_medium_threshold.

    Logica:
    - HIGH: max_score ≥ high_threshold AND n_above_threshold ≥ 2
            (almeno 2 chunks fortemente rilevanti)
    - MEDIUM: max_score ≥ medium_threshold
              (almeno 1 chunk parzialmente rilevante)
    - LOW: max_score < medium_threshold (o n_hits = 0)
    """
    if not hits:
        return GroundingMetadata(
            tier="low",
            max_score=0.0,
            mean_score=0.0,
            n_hits=0,
            n_above_threshold=0,
            explanation="no_hits",
        )

    # BUGFIX 2026-07-05: con hybrid RRF h.score è un rank-score (~0.016 max),
    # non confrontabile con le soglie cosine. Usa il cosine dense preservato
    # in metadata["dense_score"]; fallback a h.score per il path non-hybrid.
    scores = [
        float((h.metadata or {}).get("dense_score", h.score)) for h in hits
    ]
    max_score = max(scores)
    mean_score = sum(scores) / len(scores)

    high_thr = RAG.grounding_high_threshold
    med_thr = RAG.grounding_medium_threshold

    n_above_med = sum(1 for s in scores if s >= med_thr)
    n_above_high = sum(1 for s in scores if s >= high_thr)

    if max_score >= high_thr and n_above_high >= 2:
        tier = "high"
        explanation = f"max={max_score:.2f}≥{high_thr}, {n_above_high} chunks ≥{high_thr}"
    elif max_score >= med_thr:
        tier = "medium"
        explanation = f"max={max_score:.2f}≥{med_thr} but only {n_above_high} chunks ≥{high_thr}"
    else:
        tier = "low"
        explanation = f"max={max_score:.2f}<{med_thr} → insufficient grounding"

    return GroundingMetadata(
        tier=tier,
        max_score=max_score,
        mean_score=mean_score,
        n_hits=len(hits),
        n_above_threshold=n_above_med,
        explanation=explanation,
    )


# ──────────────────────────────────────────────────────────────
# Backward compat: alcuni moduli (es. llm/watcher) chiamano _embed_query
# direttamente. Lo manteniamo come thin facade.
# ──────────────────────────────────────────────────────────────
def _embed_query(query: str) -> list[float]:
    return get_embedding_provider().embed(query)


# ──────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ──────────────────────────────────────────────────────────────
def _rrf_fuse(
    dense_results: list[VectorHit],
    sparse_results: list[VectorHit],
    *,
    dense_weight: float,
    sparse_weight: float,
    k: int = 60,
) -> list[VectorHit]:
    """Combina due liste rank-ordinate via Reciprocal Rank Fusion.

    Formula: score(doc) = sum_over_lists( weight / (k + rank_in_list) )

    Pro: indipendente dalla scala dei score (BM25 vs cosine sono incomparabili).
    Si lavora solo sulle posizioni.
    """
    # Key per identificare lo "stesso" chunk in entrambe le liste.
    # Usiamo (section, source, hash(text[:200])) come fingerprint.
    def _key(h: VectorHit) -> tuple:
        return (h.section, h.source, hash(h.text[:200]))

    fused: dict[tuple, tuple[float, VectorHit]] = {}
    # BUGFIX 2026-07-05: gli score RRF sono ~1/(k+rank) → max ≈ 0.016 con k=60.
    # Le soglie di grounding (0.55/0.30) sono su scala cosine 0-1 → con hybrid
    # attivo il grounding risultava SEMPRE "low". Preserviamo il cosine dense
    # originale in metadata["dense_score"]: l'RRF resta per il RANKING, il
    # cosine resta il segnale assoluto per grounding/L1.
    dense_cosines: dict[tuple, float] = {}

    for rank, hit in enumerate(dense_results):
        k_id = _key(hit)
        score = dense_weight / (k + rank + 1)
        fused[k_id] = (score, hit)
        dense_cosines[k_id] = hit.score  # cosine similarity 0-1

    for rank, hit in enumerate(sparse_results):
        k_id = _key(hit)
        contribution = sparse_weight / (k + rank + 1)
        if k_id in fused:
            existing_score, existing_hit = fused[k_id]
            # Manteniamo l'hit dal dense (ha metadata più complete forse),
            # ma sommiamo lo score
            fused[k_id] = (existing_score + contribution, existing_hit)
        else:
            fused[k_id] = (contribution, hit)

    # Aggiorna lo score di ogni VectorHit con quello RRF
    ranked = sorted(fused.items(), key=lambda kv: kv[1][0], reverse=True)
    out: list[VectorHit] = []
    for k_id, (rrf_score, hit) in ranked:
        meta = dict(hit.metadata or {})
        if k_id in dense_cosines:
            meta["dense_score"] = dense_cosines[k_id]
        out.append(VectorHit(
            text=hit.text,
            source=hit.source,
            source_type=hit.source_type,
            section=hit.section,
            score=rrf_score,
            metadata=meta,
        ))
    return out


# ──────────────────────────────────────────────────────────────
# Fase 17 — Graph fusion
# ──────────────────────────────────────────────────────────────
def _fuse_with_graph(
    base_hits: list[VectorHit],
    graph_hits: list[VectorHit],
    *,
    graph_weight: float = 0.2,
) -> list[VectorHit]:
    """RRF-style fusion fra hits dense/sparse esistenti e hits da GraphRAG.

    Strategy:
    - I chunks già in base_hits con presenza in graph_hits ottengono uno score boost.
    - I chunks SOLO in graph_hits entrano con score = graph_weight scalato per rank.
    - Score finale: tiene il valore corrente di base_hits (RRF già normalizzato)
      sommato a un piccolo bonus dalla presenza nel grafo.
    """
    if not graph_hits:
        return base_hits

    def _key(h: VectorHit) -> tuple:
        return (h.section, h.source, hash(h.text[:200]))

    base_keys = {_key(h): h for h in base_hits}
    graph_keys = {_key(h): h for h in graph_hits}

    # 1. Boost score dei base_hits che sono anche nel grafo
    boosted: list[VectorHit] = []
    for h in base_hits:
        if _key(h) in graph_keys:
            new_score = min(1.0, h.score + graph_weight * 0.5)
            boosted.append(VectorHit(
                text=h.text, source=h.source, source_type=h.source_type,
                section=h.section, score=new_score,
                metadata={**h.metadata, "_graph_boost": True},
            ))
        else:
            boosted.append(h)

    # 2. Aggiungi i graph_hits non presenti nei base con score derivato
    new_from_graph: list[VectorHit] = []
    for rank, h in enumerate(graph_hits):
        if _key(h) in base_keys:
            continue
        # Score = graph_weight scalato per posizione (1/(1+rank))
        gscore = graph_weight / (1 + rank)
        new_from_graph.append(VectorHit(
            text=h.text, source=h.source, source_type=h.source_type,
            section=h.section, score=gscore,
            metadata={**h.metadata, "_graph_only": True},
        ))

    fused = boosted + new_from_graph
    fused.sort(key=lambda h: h.score, reverse=True)
    return fused


# ──────────────────────────────────────────────────────────────
# Dedup / threshold helpers (Fase 7.2)
# ──────────────────────────────────────────────────────────────
def _filter_by_threshold(hits: list[VectorHit], threshold: float) -> list[VectorHit]:
    return [h for h in hits if h.score >= threshold]


def _deduplicate(hits: list[VectorHit], similarity_threshold: float) -> list[VectorHit]:
    if not hits or similarity_threshold >= 1.0:
        return hits
    kept: list[VectorHit] = []
    for h in hits:
        is_dup = False
        for k in kept:
            ratio = SequenceMatcher(None, h.text[:500], k.text[:500]).ratio()
            if ratio >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(h)
    return kept


# ──────────────────────────────────────────────────────────────
# Per-section retrieval (dense + optional sparse)
# ──────────────────────────────────────────────────────────────
def _retrieve_one_section(
    section: str,
    query: str,
    query_embedding: list[float],
    top_k: int,
) -> list[VectorHit]:
    """Esegue dense + sparse per UNA sezione, applica RRF se ibrido."""
    vec_store = get_vector_store_provider()
    dense_hits = vec_store.search(section, query_embedding, top_k=top_k)

    if not RAG.hybrid_retrieval_enabled:
        return dense_hits

    sparse_store = get_sparse_store_provider()
    try:
        sparse_hits = sparse_store.search(section, query, top_k=top_k)
    except Exception as e:
        print(f"[retriever] sparse search error '{section}': {e}, dense-only")
        sparse_hits = []

    if not sparse_hits:
        return dense_hits

    return _rrf_fuse(
        dense_hits, sparse_hits,
        dense_weight=RAG.rrf_dense_weight,
        sparse_weight=RAG.rrf_sparse_weight,
        k=RAG.rrf_k,
    )


# ──────────────────────────────────────────────────────────────
# Main search API
# ──────────────────────────────────────────────────────────────
def search_with_confidence(
    query: str,
    sections: list[str],
    top_k: int | None = None,
    *,
    skip_rewriting: bool = False,
    skip_rerank: bool = False,
) -> tuple[list[VectorHit], GroundingMetadata]:
    """Pipeline completa che ritorna ANCHE la GroundingMetadata (Fase 13)."""
    hits = search(query, sections, top_k=top_k,
                  skip_rewriting=skip_rewriting, skip_rerank=skip_rerank)
    metadata = _compute_grounding(hits)
    print(f"[rag] grounding: tier={metadata.tier} max={metadata.max_score:.2f} "
          f"hits={metadata.n_hits} ({metadata.explanation})")
    return hits, metadata


# ──────────────────────────────────────────────────────────────
# Fase 14 — Corrective RAG (CRAG)
# ──────────────────────────────────────────────────────────────
def search_with_correction(
    query: str,
    sections: list[str],
    top_k: int | None = None,
    *,
    skip_rewriting: bool = False,
    skip_rerank: bool = False,
    max_retries: int = 1,
) -> tuple[list[VectorHit], GroundingMetadata, str]:
    """Self-corrective RAG pipeline:

    1. Retrieve iniziale
    2. Evaluator LLM valuta i chunks
    3. Se "partial": tenta query expansion + re-retrieve, merge results
    4. Se "insufficient": ritorna comunque (composer userà tier LOW)

    Returns:
        (hits, metadata, verdict) — verdict ∈ {sufficient, partial, insufficient}
    """
    if not query.strip() or not sections:
        meta = _compute_grounding([])
        return [], meta, "insufficient"

    # 1. Retrieve iniziale
    hits = search(query, sections, top_k=top_k,
                  skip_rewriting=skip_rewriting, skip_rerank=skip_rerank)
    meta = _compute_grounding(hits)

    if not RAG.crag_enabled:
        print(f"[crag] disabled, skipping. tier={meta.tier}")
        return hits, meta, ("sufficient" if meta.tier == "high"
                            else "partial" if meta.tier == "medium"
                            else "insufficient")

    # 2. Evaluator LLM
    try:
        from ..llm.evaluator import Evaluator
        evaluator = Evaluator()
        verdict_result = evaluator.evaluate(query, hits)
        print(f"[crag] evaluator: verdict={verdict_result.verdict} "
              f"conf={verdict_result.confidence:.2f} ({verdict_result.reason[:80]})")
    except Exception as e:
        print(f"[crag] evaluator error: {e}, fallback to grounding tier")
        return hits, meta, "partial"

    # 3. Se "partial" e abbiamo varianti, fai re-query
    if (verdict_result.verdict == "partial"
            and verdict_result.suggested_query_variants
            and max_retries > 0):
        all_hits_dict = {(h.section, h.source, h.text[:100]): h for h in hits}
        for variant in verdict_result.suggested_query_variants[:RAG.crag_max_variants]:
            try:
                # Re-retrieve con la variant. Skip rewriting (la variant è già una query)
                extra_hits = search(variant, sections, top_k=top_k,
                                    skip_rewriting=True, skip_rerank=skip_rerank)
                print(f"[crag] retry with '{variant[:60]}' → {len(extra_hits)} extra hits")
                for h in extra_hits:
                    key = (h.section, h.source, h.text[:100])
                    if key not in all_hits_dict or h.score > all_hits_dict[key].score:
                        all_hits_dict[key] = h
            except Exception as e:
                print(f"[crag] variant retry error: {e}")

        # Merge + rerank
        merged_hits = sorted(all_hits_dict.values(), key=lambda h: h.score, reverse=True)
        final_k = top_k if top_k is not None else RAG.final_top_k
        hits = merged_hits[:final_k]
        meta = _compute_grounding(hits)
        print(f"[crag] post-retry: {len(hits)} hits, tier={meta.tier} max={meta.max_score:.2f}")

    return hits, meta, verdict_result.verdict


def search(
    query: str,
    sections: list[str],
    top_k: int | None = None,
    *,
    skip_rewriting: bool = False,
    skip_rerank: bool = False,
) -> list[VectorHit]:
    """Pipeline completa: query-rewrite → hybrid retrieve → threshold → dedup → rerank → top-k.
    """
    if not sections or not query.strip():
        return []

    final_k = top_k if top_k is not None else RAG.final_top_k

    # 1. Query rewriting (opt-in).
    # Auto-skip quando llm=cloud: Claude gestisce nativamente query rumorose
    # senza bisogno di pre-rewriting (-500ms latency).
    from ..config import PROVIDERS as _PROVIDERS
    auto_skip_rewriting = _PROVIDERS.llm == "cloud"
    if not skip_rewriting and not auto_skip_rewriting:
        from .query_rewriter import rewrite_query
        rewritten = rewrite_query(query)
        if not rewritten.strip():
            return []
        effective_query = rewritten
        if rewritten != query:
            print(f"[rag] query rewritten: {query[:60]!r} → {rewritten[:60]!r}")
    else:
        effective_query = query

    # 1b. HyDE expansion (Fase 9.9, opt-in): genera risposta ipotetica per embed
    if not skip_rewriting and RAG.hyde_enabled:
        from .query_rewriter import expand_query_hyde
        expanded = expand_query_hyde(effective_query)
        if expanded != effective_query:
            print(f"[rag] hyde expanded for embedding ({len(expanded)} chars)")
        embed_query = expanded
    else:
        embed_query = effective_query

    # 2. Embed dense (per cooldown topic upstream, useremo `effective_query` per BM25)
    # Latency #11: cache LRU sull'embed. Stessa query (post-rewrite) → embed cachato.
    # In una call media hit rate ~25-30% (utente ripete o LLM rewrita uguale).
    import time as _time
    _t_embed = _time.monotonic()
    query_embedding = _cached_embed(embed_query)
    _embed_ms = (_time.monotonic() - _t_embed) * 1000

    # 3. Hybrid retrieval cross-section
    _t_retr = _time.monotonic()
    all_hits: list[VectorHit] = []
    for slug in sections:
        section_hits = _retrieve_one_section(
            section=slug,
            query=effective_query,
            query_embedding=query_embedding,
            top_k=RAG.initial_top_k,
        )
        all_hits.extend(section_hits)
    _retr_ms = (_time.monotonic() - _t_retr) * 1000

    # Sort by score (RRF o cosine)
    all_hits.sort(key=lambda h: h.score, reverse=True)
    all_hits = all_hits[: RAG.initial_top_k * 2]   # cap pre-filtering
    print(f"[rag] embed={_embed_ms:.0f}ms hybrid_retrieve={_retr_ms:.0f}ms candidates={len(all_hits)}")

    # 3b. Fase 17 — GraphRAG layer (opt-in): aggiunge chunks dal grafo.
    # Fusion via RRF leggero: i chunks già presenti vedono il loro score
    # boostato; i nuovi entrano con score "neighborhood".
    try:
        from ..config import GRAPH as _GRAPH
        if _GRAPH.enabled:
            from .graph.retriever import graph_search
            graph_hits = graph_search(effective_query, sections)
            if graph_hits:
                all_hits = _fuse_with_graph(all_hits, graph_hits,
                                             graph_weight=_GRAPH.rrf_graph_weight)
                print(f"[rag] graphRAG: +{len(graph_hits)} candidate chunks fused")
    except Exception as e:
        print(f"[rag] graphRAG layer error (non-fatal): {e}")

    # 4. Threshold filter
    if RAG.min_score_threshold > 0:
        all_hits = _filter_by_threshold(all_hits, RAG.min_score_threshold)

    # 5. Deduplication
    if RAG.dedup_similarity_threshold < 1.0:
        all_hits = _deduplicate(all_hits, RAG.dedup_similarity_threshold)

    # 6. Reranker (opt-in). Auto-skip quando top hits hanno già score alto
    # (riduce latency 300-500ms per query "facili" senza perdita qualità).
    top_score_threshold_for_skip = 0.55
    early_skip = (
        len(all_hits) >= 3
        and all(h.score >= top_score_threshold_for_skip for h in all_hits[:3])
    )
    if not skip_rerank and RAG.rerank_enabled and all_hits and not early_skip:
        rerank_provider = get_rerank_provider()
        _t_rr = _time.monotonic()
        all_hits = rerank_provider.rerank(effective_query, all_hits, top_k=final_k)
        _rr_ms = (_time.monotonic() - _t_rr) * 1000
        print(f"[rag] rerank={_rr_ms:.0f}ms top_k={len(all_hits)}")
    else:
        if early_skip:
            print(f"[rag] skip rerank: top-3 score>{top_score_threshold_for_skip}")
        all_hits = all_hits[:final_k]

    return all_hits
