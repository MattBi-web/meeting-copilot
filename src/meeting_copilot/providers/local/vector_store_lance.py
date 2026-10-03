"""LanceDB vector store provider (Fase 9.7).

Vantaggi vs ChromaDB:
- Storage Arrow-based: 3-5x più veloce in I/O
- Native support per multi-vector (es. BGE-M3 colbert) → futuro upgrade
- Disk-efficient: nessun process server, file-based come Chroma
- Schema evolution: aggiungere colonne metadata senza rebuild

Layout: una table per sezione in `kb/<section>/lance/`.
"""
from __future__ import annotations

from typing import Any

from ...kb import paths
from ..base import VectorHit


class LocalLanceDBVectorStoreProvider:
    """Implementa VectorStoreProvider via LanceDB."""

    def _db(self, section: str):
        try:
            import lancedb
        except ImportError:
            raise RuntimeError("lancedb non installato. Aggiungi a pyproject.toml o usa chromadb.")
        db_path = paths.section_dir(section) / "lance"
        db_path.mkdir(parents=True, exist_ok=True)
        return lancedb.connect(str(db_path))

    def _open_or_create_table(self, section: str, dim: int | None = None):
        """Apre la table se esiste, altrimenti la crea con uno schema vuoto.
        Lo schema viene inferito al primo write."""
        db = self._db(section)
        try:
            return db.open_table(section)
        except Exception:
            return None  # caller la creerà al primo upsert

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
        db = self._db(section)
        records = [
            {
                "id": ids[i],
                "vector": embeddings[i],
                "text": documents[i],
                "source": metadatas[i].get("source", "unknown"),
                "source_type": metadatas[i].get("source_type", "unknown"),
                "section": metadatas[i].get("section", section),
                "metadata_json": _serialize_meta(metadatas[i]),
            }
            for i in range(len(ids))
        ]
        try:
            table = db.open_table(section)
            # Append (LanceDB non ha "delete+insert" atomico, ma noi cancelliamo prima)
            table.add(records)
        except Exception:
            db.create_table(section, data=records)

    def delete_section(self, section: str) -> None:
        try:
            db = self._db(section)
            try:
                db.drop_table(section)
            except Exception:
                pass
        except Exception:
            pass

    def search(
        self,
        section: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[VectorHit]:
        lance_dir = paths.section_dir(section) / "lance"
        if not lance_dir.exists():
            return []
        try:
            db = self._db(section)
            table = db.open_table(section)
            # LanceDB API: table.search(vector).limit(k).to_list()
            results = table.search(query_embedding).limit(top_k).to_list()
        except Exception as e:
            print(f"[lancedb] search error '{section}': {e}")
            return []

        hits: list[VectorHit] = []
        for r in results:
            # LanceDB ritorna _distance per default (L2 o cosine in base alla metric)
            dist = float(r.get("_distance", 0.0))
            score = max(0.0, 1.0 - dist)
            meta_dict = _deserialize_meta(r.get("metadata_json", "{}"))
            hits.append(VectorHit(
                text=r["text"],
                source=r.get("source", "unknown"),
                source_type=r.get("source_type", "unknown"),
                section=r.get("section", section),
                score=score,
                metadata=meta_dict,
            ))
        return hits

    def count(self, section: str) -> int:
        lance_dir = paths.section_dir(section) / "lance"
        if not lance_dir.exists():
            return 0
        try:
            db = self._db(section)
            table = db.open_table(section)
            return int(table.count_rows())
        except Exception:
            return 0


def _serialize_meta(meta: dict) -> str:
    import json
    return json.dumps(
        {k: v for k, v in meta.items() if k not in ("source", "source_type", "section")},
        ensure_ascii=False,
    )


def _deserialize_meta(s: str) -> dict:
    import json
    try:
        return json.loads(s)
    except Exception:
        return {}
