"""Memory manager — API alto livello (facade).

Punto unico per:
- save_session_memories(session, scope): estrae + salva memorie post-call
- get_memories_for_scope(scope): lista memorie correnti
- search_memories(query, scopes): cerca semantically
- count_for_scope(scope): badge UI count
"""
from __future__ import annotations

from .extractor import extract_memories_from_session
from .retriever import format_memories_as_context, search_relevant_memories
from .store import MemoryRecord, MemoryStore


class MemoryManager:
    """Facade. Tutti gli accessi alla memoria passano da qui."""

    # ── Save ──
    @staticmethod
    def save_session_memories(session, scope: str) -> int:
        """A fine call, estrae memorie dalla session e le salva con embedding."""
        try:
            from ..config import MEMORY
            if not MEMORY.enabled:
                return 0
        except Exception:
            return 0

        memories = extract_memories_from_session(session, scope)
        if not memories:
            return 0

        # Embed ciascuna memory
        try:
            from ..providers import get_embedding_provider
            provider = get_embedding_provider()
            for m in memories:
                try:
                    m.embedding = provider.embed(m.text)
                except Exception as e:
                    print(f"[memory] embed error per '{m.text[:50]}': {e}")
                    m.embedding = None
        except Exception as e:
            print(f"[memory] embedding provider error: {e}")

        store = MemoryStore.instance()
        store.save_batch(memories)
        print(f"[memory] salvate {len(memories)} memorie nello scope '{scope}'")
        return len(memories)

    # ── Save explicit (utente forza una memoria) ──
    @staticmethod
    def save_explicit(
        text: str, scope: str, *,
        kind: str = "semantic",
        importance: float = 0.7,
        session_id: str = "",
    ) -> MemoryRecord:
        """Salva una memoria forzata dall'utente (es. via Cmd+Shift+M in futuro)."""
        from .store import MemoryRecord
        import uuid, time
        rec = MemoryRecord(
            id=uuid.uuid4().hex[:16],
            scope=scope,
            text=text[:500],
            kind=kind if kind in ("semantic", "episodic") else "semantic",
            importance=max(0.0, min(1.0, importance)),
            created_at=time.time(),
            session_id=session_id,
        )
        try:
            from ..providers import get_embedding_provider
            rec.embedding = get_embedding_provider().embed(rec.text)
        except Exception:
            rec.embedding = None
        MemoryStore.instance().save(rec)
        return rec

    # ── Query ──
    @staticmethod
    def search(query: str, scopes: list[str], top_k: int = 5) -> list:
        return search_relevant_memories(query, scopes, top_k=top_k)

    @staticmethod
    def format_as_context(memories: list) -> str:
        return format_memories_as_context(memories)

    @staticmethod
    def list_for_scope(scope: str, limit: int = 200) -> list[MemoryRecord]:
        return MemoryStore.instance().list_by_scope(scope, limit=limit)

    @staticmethod
    def count_for_scope(scope: str) -> int:
        store = MemoryStore.instance()
        return sum(1 for _ in store.list_by_scope(scope, limit=10000))

    @staticmethod
    def list_scopes() -> list[tuple[str, int]]:
        return MemoryStore.instance().list_scopes()

    @staticmethod
    def total_count() -> int:
        return MemoryStore.instance().count()

    # ── Maintenance ──
    @staticmethod
    def delete_by_id(mem_id: str) -> None:
        MemoryStore.instance().delete_by_id(mem_id)

    @staticmethod
    def delete_scope(scope: str) -> int:
        return MemoryStore.instance().delete_by_scope(scope)

    @staticmethod
    def purge_all() -> int:
        return MemoryStore.instance().purge_all()
