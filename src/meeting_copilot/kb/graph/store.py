"""GraphStore — SQLite-backed graph (entities, relations, mentions).

Schema:
    entities(id, scope, name, type, embedding BLOB, mention_count, created_at)
    relations(id, scope, src_id, dst_id, label, weight, evidence_chunk_id)
    mentions(entity_id, scope, chunk_id, source)
    UNIQUE (entity_id, chunk_id) per evitare doppioni.

Embedding store come BLOB (numpy float32). Search via cosine in Python — per KB
piccolo-medio (~thousands of entità) è sufficiente. Indici SQL su (scope, name)
e (scope, src_id) per accelerare lookup.

scope = slug della sezione KB (es. "acme_meeting_v6"). Ogni sezione ha il
suo sotto-grafo isolato. Multi-section query = union dei sotto-grafi.
"""
from __future__ import annotations

import sqlite3
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import section_dir


# ──────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────
@dataclass
class GraphEntity:
    id: int
    scope: str
    name: str
    type: str                      # "person" | "org" | "product" | "concept" | "other"
    embedding: list[float] = field(default_factory=list)
    mention_count: int = 0


@dataclass
class GraphRelation:
    id: int
    scope: str
    src_id: int
    dst_id: int
    label: str                     # es. "integra_con", "compete_con", "prezzo_di"
    weight: float = 1.0            # aumenta se ri-osservata in più chunks
    evidence_chunk_id: str = ""


@dataclass
class GraphMention:
    entity_id: int
    scope: str
    chunk_id: str
    source: str


# ──────────────────────────────────────────────────────────────
# Helpers BLOB embedding
# ──────────────────────────────────────────────────────────────
def _embedding_to_blob(emb: list[float]) -> bytes:
    if not emb:
        return b""
    return struct.pack(f"{len(emb)}f", *emb)


def _blob_to_embedding(blob: bytes) -> list[float]:
    if not blob:
        return []
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


# ──────────────────────────────────────────────────────────────
# GraphStore
# ──────────────────────────────────────────────────────────────
class GraphStore:
    """Wrapper SQLite. Per scope (= sezione KB) c'è 1 file `graph.db`."""

    @staticmethod
    def db_path(scope: str) -> Path:
        return section_dir(scope) / "graph.db"

    def __init__(self, scope: str):
        self.scope = scope
        self.path = GraphStore.db_path(scope)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path))

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL DEFAULT 'other',
                    embedding BLOB,
                    mention_count INTEGER DEFAULT 0,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_entities_scope_name
                    ON entities (scope, name);
                CREATE INDEX IF NOT EXISTS idx_entities_scope
                    ON entities (scope);

                CREATE TABLE IF NOT EXISTS relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL,
                    src_id INTEGER NOT NULL,
                    dst_id INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    evidence_chunk_id TEXT,
                    UNIQUE (scope, src_id, dst_id, label)
                );
                CREATE INDEX IF NOT EXISTS idx_rel_scope_src
                    ON relations (scope, src_id);
                CREATE INDEX IF NOT EXISTS idx_rel_scope_dst
                    ON relations (scope, dst_id);

                CREATE TABLE IF NOT EXISTS mentions (
                    entity_id INTEGER NOT NULL,
                    scope TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    source TEXT,
                    PRIMARY KEY (entity_id, chunk_id)
                );
                CREATE INDEX IF NOT EXISTS idx_mentions_chunk
                    ON mentions (chunk_id);
                CREATE INDEX IF NOT EXISTS idx_mentions_entity
                    ON mentions (entity_id);
                """
            )

    # ─── Reset ───
    def reset(self) -> None:
        """Wipe completo del grafo dello scope. Usato pre-rebuild."""
        with self._conn() as c:
            c.execute("DELETE FROM mentions WHERE scope = ?", (self.scope,))
            c.execute("DELETE FROM relations WHERE scope = ?", (self.scope,))
            c.execute("DELETE FROM entities WHERE scope = ?", (self.scope,))

    # ─── Entities ───
    def upsert_entity(
        self,
        name: str,
        type_: str = "other",
        embedding: list[float] | None = None,
    ) -> int:
        """Inserisce o aggiorna un'entità. Ritorna l'id. Idempotente sul name."""
        name = name.strip()
        if not name:
            return 0
        with self._conn() as c:
            row = c.execute(
                "SELECT id FROM entities WHERE scope = ? AND lower(name) = lower(?)",
                (self.scope, name),
            ).fetchone()
            if row:
                eid = row[0]
                c.execute(
                    "UPDATE entities SET mention_count = mention_count + 1 WHERE id = ?",
                    (eid,),
                )
                if embedding is not None:
                    # mantiene la prima embedding; aggiornarla solo se assente
                    cur = c.execute("SELECT embedding FROM entities WHERE id = ?",
                                    (eid,)).fetchone()
                    if not cur or not cur[0]:
                        c.execute(
                            "UPDATE entities SET embedding = ? WHERE id = ?",
                            (_embedding_to_blob(embedding), eid),
                        )
                return int(eid)
            cur = c.execute(
                "INSERT INTO entities (scope, name, type, embedding, mention_count, created_at) "
                "VALUES (?, ?, ?, ?, 1, ?)",
                (self.scope, name, type_, _embedding_to_blob(embedding or []), time.time()),
            )
            return int(cur.lastrowid or 0)

    def get_entity(self, eid: int) -> GraphEntity | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT id, scope, name, type, embedding, mention_count "
                "FROM entities WHERE id = ?",
                (eid,),
            ).fetchone()
        if not row:
            return None
        return GraphEntity(
            id=row[0], scope=row[1], name=row[2], type=row[3],
            embedding=_blob_to_embedding(row[4] or b""),
            mention_count=row[5],
        )

    def list_entities(self) -> list[GraphEntity]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, scope, name, type, embedding, mention_count "
                "FROM entities WHERE scope = ? ORDER BY mention_count DESC",
                (self.scope,),
            ).fetchall()
        return [
            GraphEntity(
                id=r[0], scope=r[1], name=r[2], type=r[3],
                embedding=_blob_to_embedding(r[4] or b""),
                mention_count=r[5],
            )
            for r in rows
        ]

    def find_entities_by_embedding(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[GraphEntity, float]]:
        """Ritorna (entity, cosine_score) ordinati desc, filtrando score>0."""
        if not query_embedding:
            return []
        entities = self.list_entities()
        scored = [(e, _cosine(e.embedding, query_embedding)) for e in entities if e.embedding]
        scored = [(e, s) for e, s in scored if s > 0]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]

    # ─── Relations ───
    def upsert_relation(
        self,
        src_id: int,
        dst_id: int,
        label: str,
        evidence_chunk_id: str = "",
    ) -> None:
        if src_id <= 0 or dst_id <= 0 or src_id == dst_id:
            return
        with self._conn() as c:
            row = c.execute(
                "SELECT id, weight FROM relations "
                "WHERE scope = ? AND src_id = ? AND dst_id = ? AND label = ?",
                (self.scope, src_id, dst_id, label),
            ).fetchone()
            if row:
                # incrementa weight (evidenza ricorrente)
                c.execute(
                    "UPDATE relations SET weight = weight + 0.5 WHERE id = ?",
                    (row[0],),
                )
            else:
                c.execute(
                    "INSERT INTO relations "
                    "(scope, src_id, dst_id, label, weight, evidence_chunk_id) "
                    "VALUES (?, ?, ?, ?, 1.0, ?)",
                    (self.scope, src_id, dst_id, label, evidence_chunk_id),
                )

    def neighbors(self, entity_id: int, hops: int = 1) -> list[int]:
        """Ritorna gli id entità raggiungibili in `hops` step (BFS).
        Esclude l'entità di partenza."""
        if hops < 1 or entity_id <= 0:
            return []
        visited: set[int] = {entity_id}
        frontier: set[int] = {entity_id}
        for _ in range(hops):
            new_frontier: set[int] = set()
            with self._conn() as c:
                for fid in frontier:
                    # out-edges
                    out_rows = c.execute(
                        "SELECT dst_id FROM relations WHERE scope = ? AND src_id = ?",
                        (self.scope, fid),
                    ).fetchall()
                    # in-edges (relazioni sono spesso semanticamente reversibili)
                    in_rows = c.execute(
                        "SELECT src_id FROM relations WHERE scope = ? AND dst_id = ?",
                        (self.scope, fid),
                    ).fetchall()
                    for r in out_rows + in_rows:
                        if r[0] not in visited:
                            new_frontier.add(r[0])
                            visited.add(r[0])
            if not new_frontier:
                break
            frontier = new_frontier
        return [v for v in visited if v != entity_id]

    # ─── Mentions ───
    def add_mention(
        self,
        entity_id: int,
        chunk_id: str,
        source: str = "",
    ) -> None:
        if entity_id <= 0 or not chunk_id:
            return
        with self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO mentions (entity_id, scope, chunk_id, source) "
                "VALUES (?, ?, ?, ?)",
                (entity_id, self.scope, chunk_id, source),
            )

    def chunks_for_entities(self, entity_ids: list[int]) -> list[str]:
        """Ritorna chunk_ids unici che menzionano almeno una delle entità.
        Ordinati per frequency (chunks più citati prima)."""
        if not entity_ids:
            return []
        placeholders = ",".join("?" * len(entity_ids))
        with self._conn() as c:
            rows = c.execute(
                f"SELECT chunk_id, COUNT(*) as freq FROM mentions "
                f"WHERE scope = ? AND entity_id IN ({placeholders}) "
                f"GROUP BY chunk_id ORDER BY freq DESC",
                (self.scope, *entity_ids),
            ).fetchall()
        return [r[0] for r in rows]

    # ─── Stats ───
    def stats(self) -> dict:
        with self._conn() as c:
            n_ent = c.execute(
                "SELECT COUNT(*) FROM entities WHERE scope = ?", (self.scope,)
            ).fetchone()[0]
            n_rel = c.execute(
                "SELECT COUNT(*) FROM relations WHERE scope = ?", (self.scope,)
            ).fetchone()[0]
            n_men = c.execute(
                "SELECT COUNT(*) FROM mentions WHERE scope = ?", (self.scope,)
            ).fetchone()[0]
        return {"entities": n_ent, "relations": n_rel, "mentions": n_men}
