"""Local vector store provider (ChromaDB persistent)."""
from __future__ import annotations

from typing import Any

from ...kb import paths
from ..base import VectorHit


class LocalVectorStoreProvider:
    """Implementa VectorStoreProvider via ChromaDB.

    Ogni sezione = una collection ChromaDB persistente in `kb/<section>/index/`.
    """

    def _get_collection(self, section: str):
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=str(paths.index_dir(section)),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        return client.get_or_create_collection(
            name=section,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        section: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        col = self._get_collection(section)
        col.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def delete_section(self, section: str) -> None:
        """Cancella la collection COMPLETAMENTE (no solo records).
        Necessario quando si cambia embedding model: la dimension è lockata
        sulla collection, non sui records → bisogna ricreare from scratch."""
        try:
            import chromadb
            from chromadb.config import Settings
            client = chromadb.PersistentClient(
                path=str(paths.index_dir(section)),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            try:
                client.delete_collection(name=section)
            except Exception:
                # Collection non esiste o errore non bloccante: ignora
                pass
        except Exception:
            pass

    def search(
        self,
        section: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[VectorHit]:
        if not paths.index_dir(section).exists():
            return []
        try:
            col = self._get_collection(section)
            n = col.count()
            if n == 0:
                return []
            res = col.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, n),
            )
        except Exception:
            return []

        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]

        hits: list[VectorHit] = []
        for _id, doc, meta, dist in zip(ids, docs, metas, distances):
            score = max(0.0, 1.0 - float(dist))  # cosine: dist = 1 - sim
            hits.append(VectorHit(
                text=doc,
                source=meta.get("source", "unknown"),
                source_type=meta.get("source_type", "unknown"),
                section=meta.get("section", section),
                score=score,
                metadata={k: v for k, v in meta.items()
                          if k not in ("source", "source_type", "section")},
            ))
        return hits

    def count(self, section: str) -> int:
        if not paths.index_dir(section).exists():
            return 0
        try:
            return self._get_collection(section).count()
        except Exception:
            return 0
