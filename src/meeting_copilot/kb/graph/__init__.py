"""GraphRAG — Fase 17.

Knowledge-graph layer leggero (SQLite + embeddings) per neighborhood expansion
del retrieval. Opt-in via GRAPH.enabled.

Architettura:
  - store.py     → SQLite: entities, relations, mentions
  - extractor.py → LLM extraction (entità + relazioni) da chunks
  - builder.py   → orchestrazione: extract → merge duplicates → store
  - retriever.py → query → entity match → 1-hop neighborhood → chunks

Nessun modulo esterno richiesto (no networkx, no neo4j). Tutto on-device.
"""
from .store import (
    GraphEntity,
    GraphRelation,
    GraphMention,
    GraphStore,
)

__all__ = [
    "GraphEntity",
    "GraphRelation",
    "GraphMention",
    "GraphStore",
]
