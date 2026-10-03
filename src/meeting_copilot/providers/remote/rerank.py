"""Remote RerankProvider — Fase 12."""
from __future__ import annotations

from ..base import VectorHit
from ._client import request_json
from .vector_store import _dict_to_hit


def _hit_to_dict(h: VectorHit) -> dict:
    return {
        "text": h.text,
        "source": h.source,
        "source_type": h.source_type,
        "section": h.section,
        "score": h.score,
        "metadata": h.metadata,
    }


class RemoteRerankProvider:
    """Implementa RerankProvider via POST /rerank.

    Nota: trasportare i full hits su HTTP ad ogni rerank è meno efficiente
    del local; per KB grandi conviene tenere reranker locale e LLM remoto.
    """

    def warmup(self) -> None:
        return None

    def rerank(
        self,
        query: str,
        hits: list[VectorHit],
        top_k: int | None = None,
    ) -> list[VectorHit]:
        if not hits:
            return []
        data = request_json("POST", "/rerank", json_body={
            "query": query,
            "hits": [_hit_to_dict(h) for h in hits],
            "top_k": top_k,
        })
        return [_dict_to_hit(h) for h in (data.get("hits") or [])]
