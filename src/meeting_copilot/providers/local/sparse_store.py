"""Local BM25 sparse store provider.

BM25 (Best Matching 25) è un algoritmo classico di information retrieval
basato su keyword frequency + inverse document frequency + normalizzazione.
È COMPLEMENTARE al dense retrieval:
- BM25 → ottimo su match esatti (nomi propri, codici prodotto, sigle, "ProductB")
- Dense → ottimo su match semantici ("quanto costa" matcha "prezzo")

Combinando i due via Reciprocal Rank Fusion (RRF) si ottiene il recall@10
da ~65-78% a ~91% confermato da letteratura production.

Persistenza: pickle del `bm25s.BM25` per sezione in `kb/<section>/bm25.pkl`.
Più documenti + lista metadati paralleli salvata accanto.

Library: `bm25s` (più veloce di rank_bm25, lazy compilation).
Fallback: se la lib non c'è, BM25 disabilitato (hybrid search degenera a dense-only).
"""
from __future__ import annotations

import json
import pickle
from typing import Any

from ...kb import paths
from ..base import VectorHit


class LocalBM25Provider:
    def __init__(self):
        self._cache: dict[str, Any] = {}  # section → (BM25 obj, ids, docs, metas)

    # ── paths ──
    def _store_path(self, section: str):
        return paths.section_dir(section) / "bm25.pkl"

    def _meta_path(self, section: str):
        return paths.section_dir(section) / "bm25_meta.json"

    # ── upsert / delete ──
    def upsert(
        self,
        section: str,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        try:
            import bm25s
        except ImportError:
            print("[bm25-local] bm25s non installato, BM25 disabilitato per questa sezione.")
            return

        paths.section_dir(section).mkdir(parents=True, exist_ok=True)

        # Tokenize + build index
        corpus_tokens = bm25s.tokenize(documents, stopwords=None)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        # Salva index + parallel arrays
        retriever.save(str(self._store_path(section).with_suffix("")))  # bm25s salva con suffix custom
        meta_payload = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        with self._meta_path(section).open("w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False)

        # Invalida cache
        self._cache.pop(section, None)

    def delete_section(self, section: str) -> None:
        for p in (self._meta_path(section),):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        # bm25s salva una cartella, lasciamo che reindex sovrascriva
        self._cache.pop(section, None)

    # ── search ──
    def _load_if_cached(self, section: str):
        if section in self._cache:
            return self._cache[section]
        meta_path = self._meta_path(section)
        if not meta_path.exists():
            return None
        try:
            import bm25s
        except ImportError:
            return None
        try:
            retriever = bm25s.BM25.load(
                str(self._store_path(section).with_suffix(""))
            )
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            payload = (retriever, meta["ids"], meta["documents"], meta["metadatas"])
            self._cache[section] = payload
            return payload
        except Exception as e:
            print(f"[bm25-local] load fallita per '{section}': {e}")
            return None

    def search(
        self,
        section: str,
        query: str,
        top_k: int = 10,
    ) -> list[VectorHit]:
        payload = self._load_if_cached(section)
        if payload is None:
            return []
        retriever, ids, documents, metadatas = payload

        try:
            import bm25s
        except ImportError:
            return []

        query_tokens = bm25s.tokenize([query], stopwords=None)
        # bm25s retrieve ritorna (results, scores) di shape (n_queries, k)
        n_avail = len(documents)
        k = min(top_k, n_avail)
        if k <= 0:
            return []
        results, scores = retriever.retrieve(query_tokens, k=k)

        # Indici dei documenti rilevanti per la prima (e unica) query
        doc_indices = results[0]
        doc_scores = scores[0]

        # Normalizza BM25 scores in [0..1] tramite max relativo della query
        max_s = float(max(doc_scores)) if len(doc_scores) > 0 else 1.0
        max_s = max(max_s, 1e-6)

        hits: list[VectorHit] = []
        for idx, raw_score in zip(doc_indices, doc_scores):
            i = int(idx)
            if i < 0 or i >= n_avail:
                continue
            meta = metadatas[i]
            hits.append(VectorHit(
                text=documents[i],
                source=meta.get("source", "unknown"),
                source_type=meta.get("source_type", "unknown"),
                section=meta.get("section", section),
                score=float(raw_score) / max_s,  # normalizzato 0..1
                metadata={k: v for k, v in meta.items()
                          if k not in ("source", "source_type", "section")},
            ))
        return hits

    def count(self, section: str) -> int:
        payload = self._load_if_cached(section)
        if payload is None:
            return 0
        return len(payload[1])  # len(ids)
