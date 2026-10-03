"""BrainStore — SQLite per la piramide di sintesi (v1.1a).

Stessa convenzione di memory/store.py: singleton, WAL,
~/.meeting_copilot/brain.db.

Tabelle:
    meetings        L1 per meeting (JSON strutturato)
    stream_digests  L2 per (user, stream), con dirty flag
    meta            L3 index card per user + chiavi varie
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

DEFAULT_DB_PATH = Path.home() / ".meeting_copilot" / "brain.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    stream      TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    ts          REAL NOT NULL,
    duration_s  INTEGER DEFAULT 0,
    l1          TEXT NOT NULL,          -- JSON: summary/decisions/actions/entities/questions
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_meetings_user_stream ON meetings (user_id, stream, ts DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_meetings_session ON meetings (user_id, session_id);

CREATE TABLE IF NOT EXISTS stream_digests (
    user_id     TEXT NOT NULL,
    stream      TEXT NOT NULL,
    digest      TEXT DEFAULT '',
    n_meetings  INTEGER DEFAULT 0,
    updated_at  REAL DEFAULT 0,
    dirty       INTEGER DEFAULT 1,
    PRIMARY KEY (user_id, stream)
);

CREATE TABLE IF NOT EXISTS meta (
    key         TEXT PRIMARY KEY,
    value       TEXT,
    updated_at  REAL
);

-- v1.1c: coda di triage tassonomia (suggerimenti che aspettano l'utente)
CREATE TABLE IF NOT EXISTS triage (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    kind        TEXT NOT NULL,           -- meeting_assignment | new_stream
    ref         TEXT NOT NULL,           -- meeting_id | JSON lista meeting_ids
    suggestion  TEXT NOT NULL,           -- JSON: {stream, confidence} | {name, rationale, members}
    status      TEXT DEFAULT 'pending',  -- pending | accepted | dismissed
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_triage_user ON triage (user_id, status);
"""


class BrainStore:
    _instance: "BrainStore | None" = None
    _instance_lock = Lock()

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._write_lock = Lock()
        self._db = sqlite3.connect(str(db_path), check_same_thread=False,
                                   timeout=5.0)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    @classmethod
    def instance(cls, db_path: Path | None = None) -> "BrainStore":
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

    # ── L1: meetings ──
    def add_meeting(self, *, user_id: str, stream: str, session_id: str,
                    ts: float, duration_s: int, l1: dict) -> str:
        mid = uuid.uuid4().hex[:16]
        with self.conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO meetings
                   (id, user_id, stream, session_id, ts, duration_s, l1, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (mid, user_id, stream, session_id, ts, duration_s,
                 json.dumps(l1, ensure_ascii=False), time.time()),
            )
            # upsert digest row + dirty
            c.execute(
                """INSERT INTO stream_digests (user_id, stream, dirty)
                   VALUES (?, ?, 1)
                   ON CONFLICT(user_id, stream)
                   DO UPDATE SET dirty = 1""",
                (user_id, stream),
            )
        return mid

    def meetings_for_stream(self, user_id: str, stream: str,
                            *, limit: int = 60) -> list[dict]:
        cur = self._db.execute(
            """SELECT id, session_id, ts, duration_s, l1 FROM meetings
               WHERE user_id = ? AND stream = ?
               ORDER BY ts DESC LIMIT ?""",
            (user_id, stream, limit),
        )
        out = []
        for mid, session_id, ts, duration_s, l1 in cur.fetchall():
            d = json.loads(l1)
            d["_id"], d["_session_id"] = mid, session_id
            d["_ts"], d["_duration_s"] = ts, duration_s
            out.append(d)
        return out

    # ── L2: digests ──
    def dirty_streams(self, user_id: str | None = None) -> list[tuple[str, str]]:
        """[(user_id, stream), ...] con dirty=1."""
        if user_id:
            cur = self._db.execute(
                "SELECT user_id, stream FROM stream_digests WHERE dirty = 1 AND user_id = ?",
                (user_id,))
        else:
            cur = self._db.execute(
                "SELECT user_id, stream FROM stream_digests WHERE dirty = 1")
        return list(cur.fetchall())

    def set_digest(self, user_id: str, stream: str, digest: str,
                   n_meetings: int) -> None:
        with self.conn() as c:
            c.execute(
                """UPDATE stream_digests
                   SET digest = ?, n_meetings = ?, updated_at = ?, dirty = 0
                   WHERE user_id = ? AND stream = ?""",
                (digest, n_meetings, time.time(), user_id, stream),
            )

    def get_digests(self, user_id: str) -> list[dict]:
        cur = self._db.execute(
            """SELECT stream, digest, n_meetings, updated_at
               FROM stream_digests WHERE user_id = ? ORDER BY updated_at DESC""",
            (user_id,))
        return [{"stream": s, "digest": d, "n_meetings": n, "updated_at": u}
                for s, d, n, u in cur.fetchall()]

    def users_with_streams(self) -> list[str]:
        cur = self._db.execute("SELECT DISTINCT user_id FROM stream_digests")
        return [r[0] for r in cur.fetchall()]

    # ── v1.1c: tassonomia / triage ──
    def reassign_meeting(self, meeting_id: str, new_stream: str) -> bool:
        """Sposta un meeting in un altro stream, marcando dirty entrambi."""
        with self.conn() as c:
            row = c.execute("SELECT user_id, stream FROM meetings WHERE id = ?",
                            (meeting_id,)).fetchone()
            if row is None:
                return False
            user_id, old_stream = row
            c.execute("UPDATE meetings SET stream = ? WHERE id = ?",
                      (new_stream, meeting_id))
            for s in (old_stream, new_stream):
                c.execute(
                    """INSERT INTO stream_digests (user_id, stream, dirty)
                       VALUES (?, ?, 1)
                       ON CONFLICT(user_id, stream) DO UPDATE SET dirty = 1""",
                    (user_id, s))
        return True

    def create_stream(self, user_id: str, stream: str) -> None:
        with self.conn() as c:
            c.execute(
                """INSERT INTO stream_digests (user_id, stream, dirty)
                   VALUES (?, ?, 1)
                   ON CONFLICT(user_id, stream) DO NOTHING""",
                (user_id, stream))

    def add_triage(self, *, user_id: str, kind: str, ref: str,
                   suggestion: dict) -> str:
        tid = uuid.uuid4().hex[:12]
        with self.conn() as c:
            c.execute(
                """INSERT INTO triage (id, user_id, kind, ref, suggestion,
                                       status, created_at)
                   VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
                (tid, user_id, kind, ref,
                 json.dumps(suggestion, ensure_ascii=False), time.time()))
        return tid

    def pending_triage(self, user_id: str) -> list[dict]:
        cur = self._db.execute(
            """SELECT id, kind, ref, suggestion, created_at FROM triage
               WHERE user_id = ? AND status = 'pending'
               ORDER BY created_at DESC""", (user_id,))
        return [{"id": i, "kind": k, "ref": r,
                 "suggestion": json.loads(s), "created_at": ts}
                for i, k, r, s, ts in cur.fetchall()]

    def resolve_triage(self, triage_id: str, status: str) -> dict | None:
        """Marca accepted/dismissed. Ritorna la riga per l'azione a valle."""
        with self.conn() as c:
            row = c.execute(
                "SELECT user_id, kind, ref, suggestion FROM triage WHERE id = ?",
                (triage_id,)).fetchone()
            if row is None:
                return None
            c.execute("UPDATE triage SET status = ? WHERE id = ?",
                      (status, triage_id))
        return {"user_id": row[0], "kind": row[1], "ref": row[2],
                "suggestion": json.loads(row[3])}

    def unassigned_meetings(self, user_id: str, *, limit: int = 200) -> list[dict]:
        return self.meetings_for_stream(user_id, "_unassigned", limit=limit)

    # ── L3: index card ──
    def set_meta(self, key: str, value: str) -> None:
        """Chiave/valore generico (token connettori, preferenze, ecc.)."""
        with self.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO meta (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, time.time()),
            )

    def get_meta(self, key: str) -> str | None:
        cur = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def set_index_card(self, user_id: str, card: str) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO meta (key, value, updated_at) VALUES (?, ?, ?)",
                (f"index_card:{user_id}", card, time.time()),
            )

    def get_index_card(self, user_id: str) -> str | None:
        cur = self._db.execute("SELECT value FROM meta WHERE key = ?",
                               (f"index_card:{user_id}",))
        row = cur.fetchone()
        return row[0] if row else None
