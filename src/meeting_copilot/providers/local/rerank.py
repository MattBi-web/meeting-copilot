"""Local reranker provider (sentence-transformers CrossEncoder)."""
from __future__ import annotations

from threading import Lock

from ...config import RAG
from ..base import VectorHit


class LocalRerankProvider:
    def __init__(self):
        self._model = None
        self._lock = Lock()
        self._load_failed = False

    def warmup(self) -> None:
        if not RAG.rerank_enabled:
            return
        self._get_model()

    def _get_model(self):
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None
        with self._lock:
            if self._model is not None:
                return self._model
            if self._load_failed:
                return None
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                print("[rerank-local] sentence-transformers non installato")
                self._load_failed = True
                return None
            try:
                print(f"[rerank-local] loading {RAG.rerank_model}…")
                self._model = CrossEncoder(
                    RAG.rerank_model,
                    max_length=512,
                    device="mps",
                )
                print(f"[rerank-local] ready on {self._model.device}")
            except Exception as e:
                print(f"[rerank-local] mps load failed: {e}, trying cpu")
                try:
                    from sentence_transformers import CrossEncoder as CE
                    self._model = CE(RAG.rerank_model, max_length=512, device="cpu")
                except Exception as e2:
                    print(f"[rerank-local] cpu fallback failed: {e2}")
                    self._load_failed = True
                    return None
        return self._model

    def rerank(
        self,
        query: str,
        hits: list[VectorHit],
        top_k: int | None = None,
    ) -> list[VectorHit]:
        if not RAG.rerank_enabled or not hits:
            return hits[:top_k] if top_k else hits
        model = self._get_model()
        if model is None:
            return hits[:top_k] if top_k else hits

        import numpy as np
        pairs = [(query, h.text) for h in hits]
        try:
            scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
            norm_scores = 1.0 / (1.0 + np.exp(-np.asarray(scores)))
        except Exception as e:
            print(f"[rerank-local] predict error: {e}")
            return hits[:top_k] if top_k else hits

        reranked = [
            VectorHit(
                text=h.text,
                source=h.source,
                source_type=h.source_type,
                section=h.section,
                score=float(s),
                metadata=h.metadata,
            )
            for h, s in zip(hits, norm_scores)
        ]
        reranked.sort(key=lambda h: h.score, reverse=True)
        return reranked[:top_k] if top_k else reranked
