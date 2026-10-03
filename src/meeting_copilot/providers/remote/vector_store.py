"""Remote VectorStoreProvider — Fase 12."""
from __future__ import annotations

from typing import Any

from ..base import VectorHit
from ._client import request_json


def _dict_to_hit(d: dict) -> VectorHit:
    return VectorHit(
        text=d.get("text", ""),
        source=d.get("source", ""),
        source_type=d.get("source_type", "unknown"),
        section=d.get("section", ""),
        score=float(d.get("score", 0.0)),
        metadata=d.get("metadata", {}),
    )


class RemoteVectorStoreProvider:
    """Implementa VectorStoreProvider via /vector/* endpoints."""

    def upsert(
        self,
        section: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        request_json("POST", "/vector/upsert", json_body={
            "section": section,
            "ids": ids,
            "embeddings": embeddings,
            "documents": documents,
            "metadatas": metadatas,
        })

    def delete_section(self, section: str) -> None:
        request_json("POST", "/vector/delete", json_body={"section": section})

    def search(
        self,
        section: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[VectorHit]:
        data = request_json("POST", "/vector/search", json_body={
            "section": section,
            "query_embedding": query_embedding,
            "top_k": top_k,
        })
        hits = data.get("hits") or []
        return [_dict_to_hit(h) for h in hits]

    def count(self, section: str) -> int:
        data = request_json("GET", "/vector/count", params={"section": section})
        return int(data.get("count", 0))
