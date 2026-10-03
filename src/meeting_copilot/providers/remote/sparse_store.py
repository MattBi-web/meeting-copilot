"""Remote SparseStoreProvider (BM25) — Fase 12."""
from __future__ import annotations

from typing import Any

from ..base import VectorHit
from ._client import request_json
from .vector_store import _dict_to_hit


class RemoteSparseStoreProvider:
    """Implementa SparseStoreProvider via /sparse/* endpoints."""

    def upsert(
        self,
        section: str,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        request_json("POST", "/sparse/upsert", json_body={
            "section": section,
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        })

    def delete_section(self, section: str) -> None:
        request_json("POST", "/sparse/delete", json_body={"section": section})

    def search(
        self,
        section: str,
        query: str,
        top_k: int = 10,
    ) -> list[VectorHit]:
        data = request_json("POST", "/sparse/search", json_body={
            "section": section,
            "query": query,
            "top_k": top_k,
        })
        hits = data.get("hits") or []
        return [_dict_to_hit(h) for h in hits]

    def count(self, section: str) -> int:
        data = request_json("GET", "/sparse/count", params={"section": section})
        return int(data.get("count", 0))
