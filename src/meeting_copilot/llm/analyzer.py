"""Analyzer agent: classifier veloce sulle ultime righe trascrizione.

Modello: llama3.2:3b (default, ~200ms per chiamata su M5).
Output strutturato JSON in 4 categorie:
- "question"            → l'interlocutore ti ha appena chiesto qualcosa
- "objection"           → ha sollevato un dubbio / contestazione / preoccupazione
- "pitch_opportunity"   → ha menzionato qualcosa su cui tu hai una risposta forte
- "noise"               → chiacchiera generica, nessuna azione richiesta

Solo le prime 3 categorie scatenano il composer; "noise" viene ignorato.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

# ollama è opzionale: usato solo se PROVIDERS.llm == "local".
# In cloud mode (Anthropic) il modulo non serve e il bundle .app lo esclude.
try:
    import ollama  # noqa: F401
except ImportError:
    ollama = None  # type: ignore[assignment]

from ..config import LLM


Category = Literal["question", "objection", "pitch_opportunity", "noise"]


@dataclass
class AnalysisResult:
    trigger: bool
    category: Category
    query: str           # frase riassuntiva, da passare al retriever
    confidence: float    # 0..1, self-stima
    raw: str             # output grezzo dell'LLM (per debug)


_SYSTEM_PROMPT = """Sei un agente che analizza in tempo reale una trascrizione di videochiamata.
Il tuo utente è un consulente che sta partecipando a una call commerciale/tecnica.
Tu non parli, osservi le ultime righe trascritte e classifichi.

Il tuo compito: classificare le ultime righe in UNA di queste 4 categorie:

1. "question"           - domanda diretta che merita risposta (con punto interrogativo
                          implicito o esplicito, anche "Mi spieghi come...", "Cosa
                          significa...", "Quanto costa...")
2. "objection"          - dubbio, preoccupazione, critica ("non sono sicuro", "ho
                          dubbi", "non funziona", "è caro", "ma...")
3. "pitch_opportunity"  - argomento menzionato dove abbiamo materiale rilevante.
                          Include: menzione di competitor, prodotto, tecnologia,
                          dato tecnico, sigle, integrazioni, casi d'uso.
4. "noise"              - SOLO se: saluti puri ("ciao", "ok perfetto"), conferme
                          minime ("sì", "no"), riempitivi puri ("allora", "uhm").

Restituisci SOLO un oggetto JSON valido:
{
  "category": "question" | "objection" | "pitch_opportunity" | "noise",
  "query": "termini chiave per retrieval, PRESERVANDO nomi propri, verbi di intent, sigle",
  "confidence": 0.0..1.0
}

═══ REGOLE ═══

- **SII RICETTIVO**: in una call commerciale tutto può essere relevante. Default
  a "pitch_opportunity" con conf 0.6+ quando vedi nomi di tecnologie/prodotti/aziende.
  Solo chiacchiera pura → "noise".

- **QUERY RICCA**: max 15 parole, ma MANTIENI nomi propri (ProductA, ContactCenterSuite, ecc.),
  verbi di intent (integrare, costare, scalare), dati (latenza, prezzo, durata).
  Es: "integrare ProductA con ContactCenterSuite" — NON "ProductA". "tempo per produzione" —
  NON "produzione".

- **TRASCRIZIONE AUTOMATICA**: può avere errori. "Genesis" → considera "ContactCenterSuite".
  "ProductA" e "ProductA" sono uguali.

- **SE PARLA L'UTENTE STESSO (non l'interlocutore)** → "noise". Indizi: tono di
  pitch, frasi tipo "vi posso dirti che", "le mostro".
"""

_USER_TEMPLATE = """Sezioni knowledge attive nella call corrente:
{sections}

Ultime righe trascritte (più recente in basso):
---
{transcript}
---

Classifica in JSON."""


class Analyzer:
    """Wrapper riusabile per l'analyzer agent."""

    def __init__(self, model: str | None = None):
        self.model = model or LLM.analyzer_model

    def analyze(
        self,
        transcript_lines: list[str],
        active_sections: list[str],
    ) -> AnalysisResult:
        """Chiamata sincrona. ~200-400ms su M5 con llama3.2:3b."""
        if not transcript_lines:
            return AnalysisResult(
                trigger=False, category="noise", query="", confidence=1.0, raw=""
            )

        sections_str = ", ".join(active_sections) if active_sections else "(nessuna)"
        transcript_str = "\n".join(transcript_lines[-LLM.analyzer_transcript_lines:])

        user_msg = _USER_TEMPLATE.format(
            sections=sections_str,
            transcript=transcript_str,
        )

        # Usa provider abstraction (supporta local/remote/cloud)
        # Quando PROVIDERS.llm=cloud → Claude Haiku 4.5 ~500ms invece di
        # Ollama local llama3.2:3b ~2-4s.
        try:
            from ..providers import get_llm_provider
            from ..providers.base import LLMMessage
            provider = get_llm_provider()
            raw = provider.complete(
                model=self.model,
                messages=[
                    LLMMessage("system", _SYSTEM_PROMPT),
                    LLMMessage("user", user_msg),
                ],
                temperature=0.2,
                max_tokens=150,
                format_json=True,
            ).strip()
        except Exception as e:
            return AnalysisResult(
                trigger=False, category="noise", query="",
                confidence=0.0, raw=f"ERROR: {e}"
            )

        # Parse JSON con fallback su regex se l'LLM mette testo extra
        parsed = _parse_json_loose(raw)
        category = parsed.get("category", "noise")
        if category not in ("question", "objection", "pitch_opportunity", "noise"):
            category = "noise"
        query = (parsed.get("query") or "").strip()
        confidence = float(parsed.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        trigger = (category != "noise") and (confidence >= LLM.analyzer_min_confidence)
        if trigger and not query:
            # Se l'analyzer non ha estratto una query, fallback: usa l'ultima riga
            query = transcript_lines[-1].strip()

        return AnalysisResult(
            trigger=trigger,
            category=category,
            query=query,
            confidence=confidence,
            raw=raw,
        )


def _parse_json_loose(s: str) -> dict:
    """Tenta JSON parse normale, poi fallback su regex per estrarre il primo {...}."""
    try:
        return json.loads(s)
    except Exception:
        pass
    match = re.search(r"\{[^{}]*\}", s, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}
