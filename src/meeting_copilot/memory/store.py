"""Memory store — SQLite + embeddings."""
from __future__ import annotations

import json
import sqlite3
import struct
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import numpy as np


DEFAULT_DB_PATH = Path.home() / ".meeting_copilot" / "memory.db"


# ──────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────
MemoryKind = Literal["episodic", "semantic"]


@dataclass
class MemoryRecord:
    id: str
    scope: str                       # es. "acme", "cliente_pharma_x"
    text: str                        # il fatto / nota
    kind: MemoryKind                 # episodic (evento specifico) | semantic (fatto generale)
    importance: float = 0.5          # 0..1 — quanto è importante (1=mai dimenticare)
    created_at: float = 0.0          # epoch
    session_id: str = ""             # origine (id sessione di trascrizione)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None  # populated on save


# ──────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    scope        TEXT NOT NULL,
    text         TEXT NOT NULL,
    kind         TEXT NOT NULL,            -- episodic | semantic
    importance   REAL DEFAULT 0.5,
    created_at   REAL NOT NULL,
    session_id   TEXT,
    metadata     TEXT,                      -- JSON
    embedding    BLOB,                      -- float32 array packed
    embedding_dim INTEGER
);

CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
"""


# ──────────────────────────────────────────────────────────────
# Embedding pack/unpack (raw float32 bytes for fast IO)
# ──────────────────────────────────────────────────────────────
def _pack_embedding(vec: list[float]) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    return arr.tobytes()


def _unpack_embedding(blob: bytes, dim: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.float32)
    if dim and arr.shape[0] != dim:
        # safety check; in produzione si potrebbero usare modelli diversi
        return arr
    return arr


# ──────────────────────────────────────────────────────────────
# Memory store
# ──────────────────────────────────────────────────────────────
class MemoryStore:
    """Singleton store. SQLite + WAL mode."""

    _instance: "MemoryStore | None" = None
    _instance_lock = Lock()

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._write_lock = Lock()
        self._db = sqlite3.connect(str(db_path), check_same_thread=False, timeout=5.0)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)
        self._migrate_bitemporal()
        self._db.commit()

    def _migrate_bitemporal(self) -> None:
        """v1.1a: validità bi-temporale (idea Zep, senza adottare Graphiti).

        valid_from     quando il fatto è diventato vero (default: created_at)
        invalid_at     quando ha smesso di esserlo (NULL = ancora valido)
        superseded_by  id della memoria che lo sostituisce
        """
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(memories)")}
        for name, ddl in (
            ("valid_from", "ALTER TABLE memories ADD COLUMN valid_from REAL"),
            ("invalid_at", "ALTER TABLE memories ADD COLUMN invalid_at REAL"),
            ("superseded_by", "ALTER TABLE memories ADD COLUMN superseded_by TEXT"),
        ):
            if name not in cols:
                self._db.execute(ddl)

    @classmethod
    def instance(cls, db_path: Path | None = None) -> "MemoryStore":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(db_path or DEFAULT_DB_PATH)
            return cls._instance

    @contextmanager
    def conn(self):
        with self._write_lock:
            try:
                yield self._db
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

    def close(self) -> None:
        with self._write_lock:
            try:
                self._db.close()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────
    # CRUD
    # ─────────────────────────────────────────────────────────
    def save(self, m: MemoryRecord) -> None:
        if not m.id:
            m.id = uuid.uuid4().hex[:16]
        if not m.created_at:
            m.created_at = time.time()
        blob = _pack_embedding(m.embedding) if m.embedding else None
        dim = len(m.embedding) if m.embedding else None
        with self.conn() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO memories(
                    id, scope, text, kind, importance, created_at,
                    session_id, metadata, embedding, embedding_dim
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    m.id, m.scope, m.text, m.kind, float(m.importance),
                    float(m.created_at), m.session_id or "",
                    json.dumps(m.metadata, ensure_ascii=False, default=str),
                    blob, dim,
                ),
            )

    def save_batch(self, memories: list[MemoryRecord]) -> int:
        for m in memories:
            self.save(m)
        return len(memories)

    def delete_by_id(self, mem_id: str) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM memories WHERE id = ?", (mem_id,))

    def delete_by_scope(self, scope: str) -> int:
        with self.conn() as c:
            cur = c.execute("DELETE FROM memories WHERE scope = ?", (scope,))
            return cur.rowcount

    def list_by_scope(self, scope: str, *, limit: int = 200) -> list[MemoryRecord]:
        cur = self._db.execute(
            """SELECT id, scope, text, kind, importance, created_at, session_id,
                      metadata, embedding, embedding_dim
               FROM memories WHERE scope = ?
               ORDER BY importance DESC, created_at DESC
               LIMIT ?""",
            (scope, limit),
        )
        return [self._row_to_record(r) for r in cur.fetchall()]

    def list_scopes(self) -> list[tuple[str, int]]:
        """Lista (scope, count) ordinati per count desc."""
        cur = self._db.execute(
            "SELECT scope, COUNT(*) AS n FROM memories GROUP BY scope ORDER BY n DESC"
        )
        return [(r[0], int(r[1])) for r in cur.fetchall()]

    def count(self) -> int:
        cur = self._db.execute("SELECT COUNT(*) FROM memories")
        return int(cur.fetchone()[0])

    def purge_all(self) -> int:
        with self.conn() as c:
            cur = c.execute("DELETE FROM memories")
            return cur.rowcount

    # ─────────────────────────────────────────────────────────
    # Similarity search via cosine (in-memory, ok per <10k memorie)
    # ─────────────────────────────────────────────────────────
    def search(
        self,
        query_embedding: list[float],
        scopes: list[str] | None = None,
        *,
        top_k: int = 5,
        min_similarity: float = 0.30,
        boost_importance: bool = True,
    ) -> list[tuple[MemoryRecord, float]]:
        """Cerca memorie per cosine similarity. Filtra per scope opzionale.

        Returns:
            Lista [(MemoryRecord, similarity_score)] ordinata desc.
        """
        if not query_embedding:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q) + 1e-9

        # v1.1a: escludi memorie superate (bi-temporale)
        if scopes:
            placeholders = ",".join("?" * len(scopes))
            sql = (
                f"""SELECT id, scope, text, kind, importance, created_at,
                           session_id, metadata, embedding, embedding_dim
                    FROM memories
                    WHERE scope IN ({placeholders}) AND embedding IS NOT NULL
                      AND invalid_at IS NULL"""
            )
            cur = self._db.execute(sql, tuple(scopes))
        else:
            cur = self._db.execute(
                """SELECT id, scope, text, kind, importance, created_at,
                          session_id, metadata, embedding, embedding_dim
                   FROM memories WHERE embedding IS NOT NULL
                     AND invalid_at IS NULL"""
            )
        rows = cur.fetchall()
        if not rows:
            return []

        results: list[tuple[MemoryRecord, float]] = []
        for r in rows:
            rec = self._row_to_record(r)
            if rec.embedding is None:
                continue
            v = np.asarray(rec.embedding, dtype=np.float32)
            v_norm = np.linalg.norm(v) + 1e-9
            sim = float(np.dot(q, v) / (q_norm * v_norm))
            if sim < min_similarity:
                continue
            # Boost per importance: importance alta → score più alto
            if boost_importance:
                sim = sim * (0.5 + 0.5 * rec.importance)
            results.append((rec, sim))

        results.sort(key=lambda kv: kv[1], reverse=True)
        return results[:top_k]

    # ─────────────────────────────────────────────────────────
    # v1.1a — bi-temporale: supersessione + dedup notturno
    # ─────────────────────────────────────────────────────────
    def mark_superseded(self, old_id: str, new_id: str) -> None:
        with self.conn() as c:
            c.execute(
                "UPDATE memories SET invalid_at = ?, superseded_by = ? WHERE id = ?",
                (time.time(), new_id, old_id),
            )

    def dedup_scope(self, scope: str, *, similarity: float = 0.92) -> int:
        """Dedup near-duplicate nello scope: il più vecchio viene marcato
        superseded dal più recente (mai cancellato — append-only, lezione
        Mem0). Chiamato dal consolidamento notturno. Ritorna n marcati."""
        cur = self._db.execute(
            """SELECT id, created_at, embedding, embedding_dim FROM memories
               WHERE scope = ? AND embedding IS NOT NULL AND invalid_at IS NULL
               ORDER BY created_at DESC LIMIT 500""",
            (scope,),
        )
        rows = cur.fetchall()
        if len(rows) < 2:
            return 0
        vecs = []
        for mem_id, created_at, blob, dim in rows:
            v = _unpack_embedding(blob, dim)
            n = np.linalg.norm(v) + 1e-9
            vecs.append((mem_id, created_at, v / n))
        superseded = 0
        invalidated: set[str] = set()
        for i in range(len(vecs)):           # i = più recente
            if vecs[i][0] in invalidated:
                continue
            for j in range(i + 1, len(vecs)):  # j = più vecchio
                if vecs[j][0] in invalidated:
                    continue
                sim = float(np.dot(vecs[i][2], vecs[j][2]))
                if sim >= similarity:
                    self.mark_superseded(vecs[j][0], vecs[i][0])
                    invalidated.add(vecs[j][0])
                    superseded += 1
        return superseded

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _row_to_record(row: tuple) -> MemoryRecord:
        (id_, scope, text, kind, importance, created_at,
         session_id, metadata_json, blob, dim) = row
        try:
            meta = json.loads(metadata_json) if metadata_json else {}
        except Exception:
            meta = {}
        embedding = None
        if blob is not None:
            arr = _unpack_embedding(blob, dim or 0)
            embedding = arr.tolist()
        return MemoryRecord(
            id=id_, scope=scope, text=text,
            kind=kind if kind in ("episodic", "semantic") else "semantic",
            importance=float(importance or 0.5),
            created_at=float(created_at or 0.0),
            session_id=session_id or "",
            metadata=meta, embedding=embedding,
        )
