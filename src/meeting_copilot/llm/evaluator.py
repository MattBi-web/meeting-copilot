"""Evaluator agent — Fase 14.1.

Riceve query + retrieved chunks. Decide se sono "sufficient" per rispondere bene,
"partial" (servono altre query expansion), o "insufficient" (KB non ha materiale).

Modello: llama3.2:3b (stesso dell'analyzer). Latency ~200-400ms.
Output strutturato JSON.

L'output guida il retriever in modalità CRAG (Corrective RAG):
- sufficient   → procedi a composer (tier high probabilmente)
- partial      → re-query con sinonimi/parafrasi (Fase 14.2)
- insufficient → fallback knowledge LLM (Fase 13 LOW tier)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

try:
    import ollama  # noqa: F401
except ImportError:
    ollama = None  # type: ignore[assignment]

from ..config import LLM


Verdict = Literal["sufficient", "partial", "insufficient"]


@dataclass(frozen=True)
class EvaluationResult:
    verdict: Verdict
    confidence: float                # 0..1
    reason: str                      # spiegazione (per debug/log)
    missing_aspects: list[str]       # aspetti della query non coperti
    suggested_query_variants: list[str]  # alternative query suggerite (max 3)
    raw: str                         # output grezzo LLM


_SYSTEM_PROMPT = """Sei un valutatore di qualità di retrieval RAG.
Ricevi una QUERY (cosa l'utente vuole sapere) e una lista di CHUNKS recuperati
dalla knowledge base. Devi decidere se i chunks contengono materiale SUFFICIENTE
per rispondere bene alla query.

CRITERI DI VALUTAZIONE:

- **sufficient**: i chunks coprono direttamente la query con dati specifici
  (numeri, nomi, processi). L'LLM composer può rispondere SOLO da loro.

- **partial**: i chunks parlano del topic ma mancano dettagli specifici
  richiesti. Potrebbe servire query expansion (sinonimi, parafrasi) per
  trovare il pezzo mancante.

- **insufficient**: i chunks NON parlano del topic, o parlano di topic
  correlato ma fuori scope. La KB non ha materiale.

OUTPUT — JSON valido, niente testo extra:
{
  "verdict": "sufficient" | "partial" | "insufficient",
  "confidence": 0.0..1.0,
  "reason": "breve spiegazione (max 30 parole)",
  "missing_aspects": ["aspetto1", "aspetto2"],     // vuota se sufficient
  "suggested_query_variants": ["variant1", "variant2"]  // SOLO se partial, max 3
}

REGOLE:
- Sii ONESTO. Se i chunks non bastano, dichiaralo. È meglio "insufficient"
  che far rispondere il composer con materiale debole.
- Per "partial": SUGGERISCI varianti di query che potrebbero trovare il missing.
  Es: query="prezzo ProductA" + chunks parlano solo di feature → suggested:
  ["costo ProductA", "pricing OurCo voicebot", "fatturazione modello"]
- Per "sufficient": missing_aspects e suggested_query_variants vuote.
"""


def _parse_json_loose(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        pass
    match = re.search(r"\{.*\}", s, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}


class Evaluator:
    """Wrapper riusabile per l'evaluator agent."""

    def __init__(self, model: str | None = None):
        # Usa lo stesso modello dell'analyzer per evitare di scaricarne un altro
        self.model = model or LLM.analyzer_model

    def evaluate(self, query: str, chunks: list) -> EvaluationResult:
        """Chiamata sincrona. ~200-500ms su llama3.2:3b.

        Args:
            query: la query (idealmente già rewritten).
            chunks: lista di VectorHit/SearchHit (anche vuota).
        """
        if not query.strip():
            return EvaluationResult(
                verdict="insufficient", confidence=1.0, reason="empty query",
                missing_aspects=[], suggested_query_variants=[], raw="",
            )

        # Formatta chunks per il LLM (top-5 max per stare nel context)
        if not chunks:
            chunks_str = "(nessun chunk recuperato)"
        else:
            parts = []
            for i, c in enumerate(chunks[:5], 1):
                snippet = (c.text or "")[:300]
                src = getattr(c, "source", "?")
                score = getattr(c, "score", 0.0)
                parts.append(f"[{i}] score={score:.2f} src={src}\n{snippet}")
            chunks_str = "\n---\n".join(parts)

        user_msg = (
            f"QUERY:\n{query}\n\n"
            f"CHUNKS RECUPERATI ({len(chunks)} totali, ne mostro fino a 5):\n"
            f"{chunks_str}\n\n"
            f"Valuta in JSON."
        )

        try:
            resp = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                format="json",
                options={
                    "temperature": 0.1,
                    "num_predict": 200,
                },
                keep_alive=LLM.ollama_keep_alive,
                stream=False,
            )
            raw = (resp.get("message", {}).get("content") or "").strip()
        except Exception as e:
            return EvaluationResult(
                verdict="partial",  # fail-soft: prova retry invece di skip
                confidence=0.0,
                reason=f"evaluator error: {e}",
                missing_aspects=[],
                suggested_query_variants=[],
                raw=f"ERROR: {e}",
            )

        parsed = _parse_json_loose(raw)

        verdict = parsed.get("verdict", "partial")
        if verdict not in ("sufficient", "partial", "insufficient"):
            verdict = "partial"

        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        reason = (parsed.get("reason") or "")[:200]
        missing = parsed.get("missing_aspects") or []
        if not isinstance(missing, list):
            missing = []

        variants = parsed.get("suggested_query_variants") or []
        if not isinstance(variants, list):
            variants = []
        variants = [str(v)[:120] for v in variants[:3] if isinstance(v, (str, int, float))]

        return EvaluationResult(
            verdict=verdict,
            confidence=confidence,
            reason=reason,
            missing_aspects=[str(m)[:80] for m in missing[:5]],
            suggested_query_variants=variants,
            raw=raw,
        )
