"""[FACADE] Reranker — wraps the configured rerank provider.

Storicamente questo modulo conteneva l'implementazione diretta del cross-encoder.
Dalla Fase 8, l'implementazione vive in `providers/local/rerank.py`. Questo file
resta solo per backward compat con import esistenti.
"""
from __future__ import annotations

from ..providers import get_rerank_provider


def warmup() -> None:
    """Backward-compat: chiama il provider warmup."""
    get_rerank_provider().warmup()


def rerank(query: str, hits: list, top_k: int | None = None) -> list:
    """Backward-compat: delega al provider."""
    return get_rerank_provider().rerank(query, hits, top_k=top_k)
