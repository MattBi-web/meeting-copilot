"""Auto-tassonomia — v1.1c (DESIGN_PRODUCT_V1_1 §3).

Hot path (all'ingest di un meeting senza stream):
    classify_meeting() → Haiku confronta l'L1 con gli stream esistenti
    (nome + prima riga del digest). Confidence ≥ AUTO_ASSIGN_THRESHOLD →
    assegnazione automatica; sotto → riga di triage per l'utente.
    Costo ~$0.001/item.

Cold path (settimanale, CLI):
    propose_streams() → clustering leggero (cosine greedy, zero dipendenze
    nuove) dei meeting non assegnati → l'LLM etichetta i cluster → PROPOSTE
    in triage. MAI creare/riorganizzare stream in silenzio.

CLI:
    python -m meeting_copilot.brain.taxonomy --propose [--user ID]
    (aggiungibile al cron dopo il consolidate, es. solo la domenica)
"""
from __future__ import annotations

import argparse
import json
import logging

from .extractor import llm_fake_enabled
from .store import BrainStore

log = logging.getLogger("meeting_copilot.brain.taxonomy")

AUTO_ASSIGN_THRESHOLD = 0.75
MIN_CLUSTER_SIZE = 2
CLUSTER_SIM_THRESHOLD = 0.60

_CLASSIFY_SYSTEM = """Sei il classificatore di Meeting Copilot.
Ricevi la sintesi di un meeting e la lista degli stream di lavoro esistenti.
Rispondi SOLO con JSON:
{"stream": "<slug dello stream più pertinente o null>",
 "confidence": 0.0-1.0,
 "rationale": "una frase"}
Se nessuno stream è pertinente: stream=null, confidence=0."""

_LABEL_SYSTEM = """Ricevi le sintesi di alcuni meeting che parlano dello stesso tema.
Proponi un nome di stream per raggrupparli. Rispondi SOLO con JSON:
{"name": "slug-breve-minuscolo", "rationale": "una frase sul tema comune"}"""


# ──────────────────────────────────────────────────────────────
# Hot path
# ──────────────────────────────────────────────────────────────
def classify_meeting(l1: dict, meeting_id: str, user_id: str,
                     *, store: BrainStore | None = None) -> dict:
    """Classifica un meeting _unassigned. Ritorna
    {"action": "assigned"|"triaged"|"skipped", "stream": ..., "confidence": ...}
    """
    store = store or BrainStore.instance()
    digests = store.get_digests(user_id)
    streams = [d for d in digests if d["stream"] != "_unassigned"]
    if not streams:
        return {"action": "skipped", "reason": "nessuno stream esistente"}

    catalog = "\n".join(
        f"- {d['stream']}: {(d['digest'].splitlines() or [''])[0][:120]}"
        for d in streams)
    summary = l1.get("summary", "")[:1200]

    if llm_fake_enabled():
        # deterministico: match se il nome stream appare nel summary
        best, conf = None, 0.0
        for d in streams:
            if d["stream"].replace("_", " ") in summary.lower():
                best, conf = d["stream"], 0.9
                break
        result = {"stream": best, "confidence": conf, "rationale": "[fake]"}
    else:
        try:
            from .coordinator import _llm_complete
            raw = _llm_complete(
                _CLASSIFY_SYSTEM,
                f"STREAM ESISTENTI:\n{catalog}\n\nMEETING:\n{summary}",
                max_tokens=150, temperature=0.0, format_json=True)
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1].removeprefix("json").strip()
            result = json.loads(cleaned)
        except Exception as e:
            log.warning("classify fallita: %s", e)
            return {"action": "skipped", "reason": str(e)}

    stream = result.get("stream")
    conf = float(result.get("confidence") or 0)
    known = {d["stream"] for d in streams}

    if stream in known and conf >= AUTO_ASSIGN_THRESHOLD:
        store.reassign_meeting(meeting_id, stream)
        log.info("auto-assign meeting %s → %s (%.2f)", meeting_id, stream, conf)
        return {"action": "assigned", "stream": stream, "confidence": conf}

    if stream in known and conf > 0.3:
        store.add_triage(user_id=user_id, kind="meeting_assignment",
                         ref=meeting_id,
                         suggestion={"stream": stream, "confidence": conf,
                                     "rationale": result.get("rationale", "")})
        return {"action": "triaged", "stream": stream, "confidence": conf}

    return {"action": "skipped", "reason": "nessun match plausibile"}


# ──────────────────────────────────────────────────────────────
# Cold path — proposte nuovi stream dai non assegnati
# ──────────────────────────────────────────────────────────────
def _embed_texts(texts: list[str]) -> list[list[float]] | None:
    if llm_fake_enabled():
        # pseudo-embedding deterministico dal primo token (per i test)
        out = []
        for t in texts:
            h = hash(t.split()[0].lower() if t.split() else t) % 997
            out.append([float((h >> i) & 1) for i in range(16)])
        return out
    try:
        from ..providers import get_embedding_provider
        return get_embedding_provider().embed_batch(texts)
    except Exception as e:
        log.warning("embedding per clustering fallito: %s", e)
        return None


def _greedy_clusters(vecs: list[list[float]],
                     threshold: float = CLUSTER_SIM_THRESHOLD) -> list[list[int]]:
    """Clustering greedy per cosine: zero dipendenze, adatto a N piccoli."""
    import numpy as np
    arr = np.asarray(vecs, dtype=float)
    norms = np.linalg.norm(arr, axis=1) + 1e-9
    arr = arr / norms[:, None]
    clusters: list[list[int]] = []
    for i in range(len(arr)):
        placed = False
        for cl in clusters:
            centroid = arr[cl].mean(axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-9)
            if float(arr[i] @ centroid) >= threshold:
                cl.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters


def propose_streams(user_id: str, *, store: BrainStore | None = None) -> int:
    """Clustering dei meeting _unassigned → proposte new_stream in triage.
    Ritorna il numero di proposte create."""
    store = store or BrainStore.instance()
    items = store.unassigned_meetings(user_id)
    # escludi quelli già in triage pending
    pending_refs = {t["ref"] for t in store.pending_triage(user_id)
                    if t["kind"] == "meeting_assignment"}
    pending_members: set[str] = set()
    for t in store.pending_triage(user_id):
        if t["kind"] == "new_stream":
            pending_members |= set(t["suggestion"].get("members", []))
    items = [m for m in items
             if m["_id"] not in pending_refs and m["_id"] not in pending_members]
    if len(items) < MIN_CLUSTER_SIZE:
        return 0

    texts = [m.get("summary", "")[:800] for m in items]
    vecs = _embed_texts(texts)
    if vecs is None:
        return 0

    n_proposals = 0
    for cluster in _greedy_clusters(vecs):
        if len(cluster) < MIN_CLUSTER_SIZE:
            continue
        member_ids = [items[i]["_id"] for i in cluster]
        sample = "\n---\n".join(texts[i][:400] for i in cluster[:5])
        if llm_fake_enabled():
            label = {"name": f"proposta-{n_proposals + 1}",
                     "rationale": "[fake] tema comune"}
        else:
            try:
                from .coordinator import _llm_complete
                raw = _llm_complete(_LABEL_SYSTEM, sample,
                                    max_tokens=120, temperature=0.2,
                                    format_json=True)
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("```")[1].removeprefix("json").strip()
                label = json.loads(cleaned)
            except Exception as e:
                log.warning("label cluster fallita: %s", e)
                continue
        store.add_triage(
            user_id=user_id, kind="new_stream",
            ref=json.dumps(member_ids),
            suggestion={"name": label.get("name", "nuovo-stream"),
                        "rationale": label.get("rationale", ""),
                        "members": member_ids})
        n_proposals += 1
    log.info("propose_streams user=%s: %d proposte", user_id, n_proposals)
    return n_proposals


# ──────────────────────────────────────────────────────────────
# Azioni di triage (chiamate dalla dashboard)
# ──────────────────────────────────────────────────────────────
def apply_triage(triage_id: str, action: str, *,
                 store: BrainStore | None = None,
                 override_stream: str | None = None) -> dict:
    """action: accept | dismiss. Ritorna esito per la UI."""
    store = store or BrainStore.instance()
    row = store.resolve_triage(triage_id,
                               "accepted" if action == "accept" else "dismissed")
    if row is None:
        return {"ok": False, "error": "triage non trovato"}
    if action != "accept":
        return {"ok": True, "action": "dismissed"}

    if row["kind"] == "meeting_assignment":
        stream = override_stream or row["suggestion"].get("stream")
        ok = store.reassign_meeting(row["ref"], stream)
        return {"ok": ok, "action": "assigned", "stream": stream}

    if row["kind"] == "new_stream":
        name = override_stream or row["suggestion"].get("name", "nuovo-stream")
        store.create_stream(row["user_id"], name)
        moved = 0
        for mid in row["suggestion"].get("members", []):
            moved += 1 if store.reassign_meeting(mid, name) else 0
        return {"ok": True, "action": "stream_created", "stream": name,
                "moved": moved}

    return {"ok": False, "error": f"kind sconosciuto: {row['kind']}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--propose", action="store_true",
                    help="cold path: proponi nuovi stream dai non assegnati")
    ap.add_argument("--user", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    store = BrainStore.instance()
    users = [args.user] if args.user else store.users_with_streams()
    if args.propose:
        for uid in users:
            n = propose_streams(uid, store=store)
            print(f"user {uid}: {n} proposte in triage")


if __name__ == "__main__":
    main()
