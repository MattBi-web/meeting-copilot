"""Summarizer: genera sintesi post-call in markdown a partire da una Session.

Usa lo stesso modello del composer (qwen2.5:7b di default). Una sola chiamata
finale, ~3-6 secondi.

Output: file .md con 5 sezioni:
1. Riassunto della conversazione
2. Action items (chi fa cosa, deadline se menzionata)
3. Decisioni prese
4. Open questions / follow-up
5. Fonti KB consultate
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

try:
    import ollama  # noqa: F401
except ImportError:
    ollama = None  # type: ignore[assignment]

from ..config import LLM, PROJECT_ROOT
from ..session import Session


SUMMARIES_DIR = PROJECT_ROOT / "sessions"


_SYSTEM_PROMPT = """Sei un assistente che produce sintesi professionali di videochiamate.
Riceverai la trascrizione di una call (con timestamp), eventuali appunti scritti dal partecipante,
e i suggerimenti generati durante la call. Tu devi produrre un MARKDOWN strutturato e leggibile.

REGOLE:
- Lingua: ITALIANO.
- Tono: professionale ma concreto. NIENTE preamboli tipo "Certo, ecco la sintesi".
- Vai dritto al punto.
- Se una sezione non ha contenuto reale, scrivi "_Nessun elemento rilevante._"
- Usa nomi propri/sigle/numeri esatti che trovi nella trascrizione (non riassumere a "qualcuno", "una somma").
- Per gli action items, formato: "**[Owner]**: descrizione (deadline)" quando l'info c'è.
- Mai inventare dati che non sono nella trascrizione.

STRUTTURA DEL DOCUMENTO (esatta, in questo ordine):

# Sintesi call · {DATE}

## Riassunto
2-4 paragrafi che spiegano di cosa si è parlato e come è andata.

## Action Items
Bullet list. Ogni voce: chi fa cosa, entro quando se specificato.

## Decisioni prese
Bullet list di decisioni esplicite emerse dalla conversazione.

## Open questions / follow-up
Bullet list di domande lasciate aperte o cose da approfondire in una prossima call.

## Fonti KB consultate
Bullet list dei chunks/sezioni di knowledge che sono stati pescati durante la call.
"""


def _build_user_message(session: Session) -> str:
    transcript = session.full_transcript_text()
    notes = session.notes.strip() or "(nessun appunto scritto)"

    suggestions_text = ""
    if session.suggestions:
        parts = []
        for s in session.suggestions:
            parts.append(
                f"- [{s.timestamp}] ({s.category}, query: {s.query!r})\n"
                f"  → {s.text}\n"
                f"  fonti: {', '.join(s.sources) if s.sources else '(nessuna)'}"
            )
        suggestions_text = "\n".join(parts)
    else:
        suggestions_text = "(nessun suggerimento generato)"

    sources_used = session.all_sources_cited() or ["(nessuna)"]

    return f"""## Trascrizione della call
{transcript}

## Appunti scritti dall'utente
{notes}

## Suggerimenti generati durante la call
{suggestions_text}

## Sezioni KB attive all'avvio
{', '.join(session.initial_active_sections) or '(nessuna)'}

## Sorgenti KB consultate via suggerimenti
{chr(10).join(f"- {s}" for s in sources_used)}

Produci ora il markdown della sintesi seguendo la struttura indicata.
"""


def summarize_session(session: Session, model: str | None = None) -> Path:
    """Genera il markdown di sintesi e lo salva su disco.

    Returns:
        Path al file .md generato.
    """
    model = model or LLM.composer_model

    # Non chiamare se non c'è abbastanza materiale
    if not session.transcript_lines and not session.notes:
        # Genera comunque uno stub minimo
        out_path = SUMMARIES_DIR / f"summary_{session.id}.md"
        out_path.write_text(
            f"# Sintesi call · {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
            "_Sessione vuota: nessuna trascrizione né appunti._\n",
            encoding="utf-8",
        )
        return out_path

    user_msg = _build_user_message(session)

    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        options={
            "temperature": 0.3,
            "num_predict": 1200,    # cap generoso per riassunti lunghi
        },
        keep_alive=LLM.ollama_keep_alive,
        stream=False,
    )
    md = (resp.get("message", {}).get("content") or "").strip()

    if not md.startswith("#"):
        # Fallback header se l'LLM non l'ha messo
        md = f"# Sintesi call · {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n" + md

    out_path = SUMMARIES_DIR / f"summary_{session.id}.md"
    out_path.write_text(md, encoding="utf-8")
    return out_path
