"""Consolidamento notturno — L2 (digest per stream) + L3 (index card).

"Sleep-time compute": MAI nel request path. Lancio:
    python -m meeting_copilot.brain.consolidate            # solo stream dirty
    python -m meeting_copilot.brain.consolidate --all      # forza tutto
    python -m meeting_copilot.brain.consolidate --user ID  # un utente

Cron (Mac dev):   crontab: 30 3 * * * cd <repo> && .venv/bin/python -m meeting_copilot.brain.consolidate
Fly.io (prod):    scheduled machine notturna (stesso comando).

Strategia costi: digest L2 con Haiku (~$0.01-0.02/stream/notte, solo dirty);
index card L3 è TEMPLATE deterministico (zero LLM) — una riga per stream
presa dalla prima frase del digest + statistiche. Dedup memorie episodiche
best-effort sui scope toccati (bi-temporale, vedi memory/store.py).
"""
from __future__ import annotations

import argparse
import logging
import time

from .store import BrainStore

log = logging.getLogger("meeting_copilot.brain.consolidate")

_DIGEST_SYSTEM = """Sei il consolidatore notturno di Meeting Copilot.
Ricevi le sintesi (L1) degli ultimi meeting di UNO stream di lavoro.
Produci un DIGEST dello stream in italiano, max 500 token, formato:

STATO: 1-2 frasi su dove è il progetto ora.
DECISIONI CHIAVE: bullet (le più recenti prevalgono su quelle superate).
AZIONI APERTE: bullet con owner se noto.
PERSONE/ENTITÀ: elenco compatto.
QUESTIONI APERTE: bullet.

Sintetizza ATTRAVERSO i meeting (non meeting per meeting). Le informazioni
più recenti sostituiscono le più vecchie in caso di conflitto."""


def _digest_via_llm(stream: str, meetings: list[dict]) -> str | None:
    from .extractor import llm_fake_enabled
    if llm_fake_enabled():
        n = len(meetings)
        last = meetings[0].get("summary", "") if meetings else ""
        return (f"STATO: [fake digest] {n} meeting nello stream {stream}. "
                f"Ultimo: {last[:80]}\nDECISIONI CHIAVE:\n• [fake]\n"
                f"AZIONI APERTE:\n• [fake]")
    try:
        from ..providers import get_llm_provider
        from ..providers.base import LLMMessage
        from ..config import LLM
        # dal più recente; cap ~15K token di input
        blocks, budget = [], 15000 * 4  # ~4 char/token
        for m in meetings:
            block = (f"--- meeting {time.strftime('%Y-%m-%d', time.localtime(m['_ts']))} ---\n"
                     f"{m.get('summary', '')}\n"
                     f"decisioni: {m.get('decisions', [])}\n"
                     f"azioni: {m.get('action_items', [])}\n"
                     f"questioni: {m.get('open_questions', [])}")
            budget -= len(block)
            if budget < 0:
                break
            blocks.append(block)
        provider = get_llm_provider()
        model = getattr(LLM, "analyzer_model", None) or getattr(
            LLM, "composer_model", "qwen2.5:7b-instruct")
        return provider.complete(
            model=model,
            messages=[LLMMessage(role="system", content=_DIGEST_SYSTEM),
                      LLMMessage(role="user",
                                 content=f"Stream: {stream}\n\n" + "\n\n".join(blocks))],
            temperature=0.2, max_tokens=700,
        )
    except Exception as e:
        log.warning("digest LLM fallito per %s: %s", stream, e)
        return None


def _build_index_card(user_id: str, store: BrainStore) -> str:
    """L3 deterministico: una riga per stream + stats. Zero LLM."""
    digests = store.get_digests(user_id)
    if not digests:
        return "Nessuno stream attivo."
    lines = [f"INDEX — {len(digests)} stream · aggiornato "
             f"{time.strftime('%Y-%m-%d %H:%M')}"]
    for d in digests:
        # prendi la riga di STATO (il LLM a volte antepone un titolo markdown)
        first = ""
        stato_next = False
        for raw in (d["digest"] or "").splitlines():
            line = raw.strip().lstrip("#*").strip()
            if stato_next and line:
                first = line
                break
            if line.upper().startswith("STATO"):
                rest = line.split(":", 1)[-1].strip().strip("*")
                if rest:
                    first = rest
                    break
                stato_next = True  # "STATO:" da solo → prendi la riga dopo
        if not first:
            first = ((d["digest"] or "").strip().splitlines() or ["(digest non ancora generato)"])[0]
        lines.append(f"• {d['stream']} — {d['n_meetings']} meeting — {first[:160]}")
    return "\n".join(lines)


def consolidate(*, user_id: str | None = None, force_all: bool = False,
                store: BrainStore | None = None) -> dict:
    """Rigenera digest dirty (o tutti con force_all) + index card. Ritorna stats."""
    store = store or BrainStore.instance()

    if force_all:
        targets = [(u, d["stream"]) for u in
                   ([user_id] if user_id else store.users_with_streams())
                   for d in store.get_digests(u)]
    else:
        targets = store.dirty_streams(user_id)

    n_ok, n_fail = 0, 0
    touched_users: set[str] = set()
    for uid, stream in targets:
        meetings = store.meetings_for_stream(uid, stream)
        digest = _digest_via_llm(stream, meetings)
        if digest is None:
            n_fail += 1
            continue
        store.set_digest(uid, stream, digest, n_meetings=len(meetings))
        touched_users.add(uid)
        n_ok += 1
        log.info("digest ok: user=%s stream=%s (%d meeting)", uid, stream, len(meetings))

    # L3 per gli utenti toccati (o tutti con force_all)
    users = touched_users or set([user_id] if user_id else store.users_with_streams())
    for uid in users:
        store.set_index_card(uid, _build_index_card(uid, store))

    # Dedup memorie episodiche best-effort (bi-temporale) sugli scope toccati
    n_superseded = 0
    try:
        from ..memory.store import MemoryStore
        mem = MemoryStore.instance()
        if hasattr(mem, "dedup_scope"):
            for uid, stream in targets:
                n_superseded += mem.dedup_scope(stream)
    except Exception as e:
        log.debug("dedup memorie saltato: %s", e)

    stats = {"digests_ok": n_ok, "digests_failed": n_fail,
             "index_cards": len(users), "memories_superseded": n_superseded}
    log.info("consolidate: %s", stats)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="forza tutti gli stream")
    ap.add_argument("--user", default=None, help="solo questo user_id")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG,
                        format="%(name)s: %(message)s")
    stats = consolidate(user_id=args.user, force_all=args.all)
    print(stats)


if __name__ == "__main__":
    main()
