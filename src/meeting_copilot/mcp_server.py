"""meeting_copilot_mcp — MCP server sul meeting brain (v1.2, DESIGN_PRODUCT_V1_1 §2b).

Espone la piramide di sintesi (stream, digest, meeting, azioni) e il
Coordinator come tool MCP. L'utente lo collega a Claude Desktop / Cowork
(stdio) e il suo assistente interroga il suo meeting brain.

Setup Claude Desktop (~/Library/Application Support/Claude/claude_desktop_config.json):
    {"mcpServers": {"meeting-copilot": {
        "command": "/percorso/repo/.venv/bin/python",
        "args": ["-m", "meeting_copilot.mcp_server"],
        "env": {"PYTHONPATH": "/percorso/repo/src"}}}}

Trasporto: stdio (locale). Remoto (streamable HTTP + OAuth 2.1 per i
connettori claude.ai) → v2, quando il server è multi-tenant in produzione.

Note di design:
- Tutti i tool sono READ-ONLY sul brain (nessuna scrittura via MCP in v1.2).
- `ask_coordinator` è l'unico che spende token nostri (router+risposta);
  gli altri leggono SQLite e costano zero — Claude può comporli da solo.
- Single-user: usa lo stesso uid anonimo della dashboard.
"""
from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:
    raise RuntimeError(
        "SDK MCP non installato: uv pip install 'mcp[cli]'") from e

mcp = FastMCP("meeting_copilot_mcp")

_UID = "00000000-0000-0000-0000-000000000000"  # single-user (come dashboard)


def _store():
    from meeting_copilot.brain.store import BrainStore
    return BrainStore.instance()


def _err(msg: str, hint: str = "") -> str:
    return json.dumps({"error": msg, "hint": hint}, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────
# Tools — lettura piramide (zero costo LLM)
# ──────────────────────────────────────────────────────────────
@mcp.tool(
    name="mc_list_streams",
    annotations={"title": "Elenca gli stream di lavoro",
                 "readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
async def mc_list_streams() -> str:
    """Elenca gli stream (progetti/clienti) del meeting brain con statistiche.

    Ritorna per ogni stream: nome, numero di meeting, ultimo aggiornamento e
    la prima riga dello stato. Usalo come punto di partenza per capire su
    cosa sta lavorando l'utente. Per il dettaglio usa mc_get_stream_digest.

    Returns:
        str: JSON {"streams": [{"stream", "n_meetings", "updated_at", "stato"}]}
    """
    digests = _store().get_digests(_UID)
    out = []
    for d in digests:
        if d["stream"] == "_unassigned":
            continue
        first = (d["digest"].splitlines() or [""])[0]
        out.append({"stream": d["stream"], "n_meetings": d["n_meetings"],
                    "updated_at": d["updated_at"],
                    "stato": first.removeprefix("STATO:").strip()[:200]})
    if not out:
        return _err("nessuno stream trovato",
                    "l'utente non ha ancora call consolidate: serve almeno "
                    "una call + `python -m meeting_copilot.brain.consolidate`")
    return json.dumps({"streams": out}, ensure_ascii=False)


class StreamInput(BaseModel):
    """Input per operazioni su un singolo stream."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    stream: str = Field(..., min_length=1, max_length=100,
                        description="Slug dello stream (da mc_list_streams), "
                                    "es. 'acme_demo'")


@mcp.tool(
    name="mc_get_stream_digest",
    annotations={"title": "Digest completo di uno stream",
                 "readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
async def mc_get_stream_digest(params: StreamInput) -> str:
    """Ritorna il digest completo di uno stream: stato, decisioni chiave,
    azioni aperte, persone/entità, questioni aperte — sintetizzato attraverso
    tutti i meeting dello stream dal consolidamento notturno.

    Args:
        params: stream — slug dello stream.

    Returns:
        str: JSON {"stream", "n_meetings", "digest"} oppure {"error", "hint"}
    """
    for d in _store().get_digests(_UID):
        if d["stream"] == params.stream:
            return json.dumps({"stream": d["stream"],
                               "n_meetings": d["n_meetings"],
                               "digest": d["digest"]}, ensure_ascii=False)
    return _err(f"stream '{params.stream}' non trovato",
                "usa mc_list_streams per i nomi validi")


class MeetingsInput(BaseModel):
    """Input per la lista meeting di uno stream."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    stream: str = Field(..., min_length=1, max_length=100,
                        description="Slug dello stream")
    limit: int = Field(default=10, ge=1, le=50,
                       description="Quanti meeting (dal più recente)")


@mcp.tool(
    name="mc_list_meetings",
    annotations={"title": "Sintesi dei meeting di uno stream",
                 "readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
async def mc_list_meetings(params: MeetingsInput) -> str:
    """Ritorna le sintesi L1 dei meeting recenti di uno stream: summary,
    decisioni, action items, entità, domande aperte, con data e durata.
    Più granulare del digest — usalo per dettagli su call specifiche.

    Args:
        params: stream + limit.

    Returns:
        str: JSON {"stream", "meetings": [{...L1, "date", "duration_s"}]}
    """
    import time as _t
    meetings = _store().meetings_for_stream(_UID, params.stream,
                                            limit=params.limit)
    if not meetings:
        return _err(f"nessun meeting nello stream '{params.stream}'",
                    "usa mc_list_streams per i nomi validi")
    out = []
    for m in meetings:
        out.append({
            "date": _t.strftime("%Y-%m-%d %H:%M", _t.localtime(m["_ts"])),
            "duration_s": m["_duration_s"],
            "summary": m.get("summary", ""),
            "decisions": m.get("decisions", []),
            "action_items": m.get("action_items", []),
            "entities": m.get("entities", []),
            "open_questions": m.get("open_questions", []),
        })
    return json.dumps({"stream": params.stream, "meetings": out},
                      ensure_ascii=False)


class SearchInput(BaseModel):
    """Input per la ricerca full-text nei meeting."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., min_length=2, max_length=300,
                       description="Testo da cercare (case-insensitive) in "
                                   "summary/decisioni/azioni di tutti i meeting")
    stream: Optional[str] = Field(default=None,
                                  description="Limita a uno stream (opzionale)")
    limit: int = Field(default=8, ge=1, le=25)


@mcp.tool(
    name="mc_search_meetings",
    annotations={"title": "Cerca nei meeting",
                 "readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
async def mc_search_meetings(params: SearchInput) -> str:
    """Cerca un testo in tutte le sintesi meeting (summary, decisioni, azioni,
    entità). Match semplice case-insensitive — per domande semantiche complesse
    usa mc_ask_coordinator.

    Args:
        params: query + stream opzionale + limit.

    Returns:
        str: JSON {"matches": [{"stream", "date", "summary", "matched_in"}]}
    """
    import time as _t
    store = _store()
    streams = ([params.stream] if params.stream
               else [d["stream"] for d in store.get_digests(_UID)])
    q = params.query.lower()
    matches = []
    for s in streams:
        for m in store.meetings_for_stream(_UID, s, limit=100):
            hay = {
                "summary": m.get("summary", ""),
                "decisions": " ".join(m.get("decisions", [])),
                "actions": " ".join(a.get("text", "") if isinstance(a, dict)
                                    else str(a) for a in m.get("action_items", [])),
                "entities": " ".join(m.get("entities", [])),
            }
            hit_in = [k for k, v in hay.items() if q in v.lower()]
            if hit_in:
                matches.append({
                    "stream": s,
                    "date": _t.strftime("%Y-%m-%d", _t.localtime(m["_ts"])),
                    "summary": m.get("summary", "")[:300],
                    "matched_in": hit_in,
                })
            if len(matches) >= params.limit:
                break
        if len(matches) >= params.limit:
            break
    if not matches:
        return _err(f"nessun match per '{params.query}'",
                    "prova termini diversi o mc_ask_coordinator per ricerca semantica")
    return json.dumps({"matches": matches}, ensure_ascii=False)


class ActionsInput(BaseModel):
    """Input per l'estrazione delle azioni aperte."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    stream: Optional[str] = Field(default=None,
                                  description="Limita a uno stream (opzionale, "
                                              "default: tutti)")


@mcp.tool(
    name="mc_get_action_items",
    annotations={"title": "Azioni aperte cross-stream",
                 "readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
async def mc_get_action_items(params: ActionsInput) -> str:
    """Ritorna gli action items dai meeting recenti, raggruppati per stream,
    con owner e data della call in cui sono emersi. Ideale per "cosa ho in
    sospeso?" / "che azioni sono aperte sul progetto X?".

    Args:
        params: stream opzionale.

    Returns:
        str: JSON {"actions": {stream: [{"text", "owner", "date"}]}}
    """
    import time as _t
    store = _store()
    streams = ([params.stream] if params.stream
               else [d["stream"] for d in store.get_digests(_UID)
                     if d["stream"] != "_unassigned"])
    out: dict = {}
    for s in streams:
        rows = []
        for m in store.meetings_for_stream(_UID, s, limit=10):
            date = _t.strftime("%Y-%m-%d", _t.localtime(m["_ts"]))
            for a in m.get("action_items", []):
                if isinstance(a, dict) and a.get("text"):
                    rows.append({"text": a["text"], "owner": a.get("owner"),
                                 "date": date})
                elif isinstance(a, str) and a:
                    rows.append({"text": a, "owner": None, "date": date})
        if rows:
            out[s] = rows[:12]
    if not out:
        return _err("nessuna azione trovata",
                    "gli action items arrivano dalle sintesi L1 post-call")
    return json.dumps({"actions": out}, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────
# Tool — Coordinator (usa i NOSTRI token: router + risposta)
# ──────────────────────────────────────────────────────────────
class AskInput(BaseModel):
    """Input per la domanda al Coordinator."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    question: str = Field(..., min_length=3, max_length=1000,
                          description="Domanda in linguaggio naturale sul "
                                      "lavoro dell'utente")
    scope: str = Field(default="everywhere",
                       description="'everywhere' oppure 'stream:<slug>'")


@mcp.tool(
    name="mc_ask_coordinator",
    annotations={"title": "Chiedi al Coordinator (sintesi ragionata)",
                 "readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": False, "openWorldHint": False},
)
async def mc_ask_coordinator(params: AskInput) -> str:
    """Fa una domanda al Coordinator di Meeting Copilot: router + sintesi
    ragionata su digest, meeting e knowledge base, con fonti citate.

    NOTA: questo tool invoca l'LLM di Meeting Copilot (costo per l'utente).
    Se puoi rispondere componendo i tool di lettura (mc_list_streams,
    mc_get_stream_digest, mc_list_meetings, mc_search_meetings,
    mc_get_action_items), preferisci quelli: sono gratuiti e più veloci.

    Args:
        params: question + scope.

    Returns:
        str: JSON {"answer", "strategy", "streams", "sources"}
    """
    from meeting_copilot.brain.coordinator import ask
    import asyncio
    result = await asyncio.to_thread(
        ask, params.question, user_id=_UID, scope=params.scope)
    return json.dumps({k: result[k] for k in
                       ("answer", "strategy", "streams", "sources")},
                      ensure_ascii=False)


def main() -> None:
    """Default: stdio (Claude Desktop). Con --http: transport HTTP per
    connettori MCP remoti (ChatGPT, deploy cloud). Porta: --port (default 8788)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true",
                    help="esponi via streamable-http invece che stdio")
    ap.add_argument("--port", type=int, default=8788)
    args = ap.parse_args()
    if args.http:
        try:
            mcp.settings.port = args.port
            mcp.run(transport="streamable-http")
            return
        except Exception as e:  # versione mcp senza supporto http
            print(f"[mcp] transport http non disponibile ({e}), fallback stdio")
    mcp.run()  # stdio


if __name__ == "__main__":
    main()
