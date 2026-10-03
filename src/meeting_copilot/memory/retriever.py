"""Memory retriever — Fase 15.3.

Cerca memorie rilevanti per la conversazione corrente. Le memorie vengono
poi aggiunte ai chunks RAG per dare al composer un contesto progressivo
(non solo KB statica, ma anche storia delle call precedenti).
"""
from __future__ import annotations

from .store import MemoryRecord, MemoryStore


def search_relevant_memories(
    query: str,
    scopes: list[str],
    *,
    top_k: int = 5,
    min_similarity: float = 0.30,
) -> list[tuple[MemoryRecord, float]]:
    """Cerca memorie rilevanti via embedding similarity sulla query.

    Args:
        query: testo della query corrente (es. ultime righe trascrizione)
        scopes: lista di scope da considerare (di solito = sezioni KB attive)
        top_k: max memorie da ritornare
        min_similarity: soglia minima

    Returns:
        Lista [(MemoryRecord, similarity)] ordinata desc.
    """
    if not query.strip() or not scopes:
        return []

    # Embed la query con stesso provider del RAG (bge-m3)
    try:
        from ..providers import get_embedding_provider
        embedding = get_embedding_provider().embed(query)
    except Exception as e:
        print(f"[memory] embed error: {e}")
        return []

    store = MemoryStore.instance()
    return store.search(
        query_embedding=embedding,
        scopes=scopes,
        top_k=top_k,
        min_similarity=min_similarity,
        boost_importance=True,
    )


def format_memories_as_context(memories: list[tuple[MemoryRecord, float]]) -> str:
    """Formatta le memorie come blocco testo per il composer prompt."""
    if not memories:
        return ""
    parts = ["═══ MEMORIE RILEVANTI DA CALL PRECEDENTI ═══"]
    for rec, score in memories:
        marker = "📌" if rec.kind == "semantic" else "🗓"
        parts.append(
            f"{marker} [{rec.kind}, importanza {rec.importance:.1f}, sim {score:.2f}] "
            f"{rec.text}"
        )
    parts.append("═══════════════════════════════════════════")
    return "\n".join(parts)
