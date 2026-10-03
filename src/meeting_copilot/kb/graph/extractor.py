"""Extractor — Fase 17.2.

Estrae entità e relazioni da un chunk via LLM con output JSON.

Input  : testo del chunk (~150-300 token).
Output : ExtractionResult con entities + relations.

Prompt design:
- Force format=json
- Vocabolario di tipi entità FISSO (person, org, product, concept, location, other).
- Vocabolario di label relazioni FISSO ridotto (10 tipi) per evitare esplosione.
- Bias verso entità DOMINIO-RELEVANT (escludendo "io", "tu", articoli, ecc.).

Costo: 1 LLM call per chunk (~200-400ms su llama3.2:3b).
Batch processing fattibile in builder.py via thread pool.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

try:
    import ollama  # noqa: F401
except ImportError:
    ollama = None  # type: ignore[assignment]

from ...config import GRAPH, LLM


EntityType = Literal["person", "org", "product", "concept", "location", "other"]

# Label canoniche per le relazioni — vocabolario chiuso per consistenza.
ALLOWED_RELATION_LABELS = (
    "integra_con",      # X integrates with Y
    "compete_con",      # X competes with Y
    "fornito_da",       # X is provided by Y
    "parte_di",         # X is part of Y
    "usa",              # X uses Y
    "prezzo_di",        # price of X
    "alternativa_a",    # X is alternative to Y
    "richiede",         # X requires Y
    "esempio_di",       # X is example of Y
    "correlato_a",      # X correlated with Y (fallback)
)


@dataclass
class ExtractedEntity:
    name: str
    type: EntityType


@dataclass
class ExtractedRelation:
    src: str            # entity name
    dst: str            # entity name
    label: str          # canonicalizzato


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    raw: str = ""


_SYSTEM_PROMPT = """Sei un extractor di knowledge graph dal testo italiano/inglese.
Ricevi un chunk e devi estrarre:
1. ENTITÀ DOMINIO-RILEVANTI: persone, organizzazioni, prodotti, concetti tecnici, luoghi.
2. RELAZIONI tra entità citate nello stesso chunk.

REGOLE PER LE ENTITÀ:
- Includi: nomi propri (es. "ProductA", "ContactCenterSuite", "Acme Consulting"), prodotti software
  e tecnologici (es. "voicebot", "RAG", "Whisper"), concetti chiave del dominio
  (es. "scalabilità", "GDPR", "latenza").
- ESCLUDI: pronomi ("io", "tu", "noi"), generici ("cosa", "modo", "tempo"),
  date e numeri puri, verbi.
- Tipo: usa ESATTAMENTE uno tra: person | org | product | concept | location | other
- Normalizza: usa la forma CANONICA dell'entità (es. "ProductA" → "ProductA"),
  ma mantieni la capitalizzazione corretta dei nomi propri.

REGOLE PER LE RELAZIONI:
- Estrai SOLO se entrambe le entità sono nella lista entities sopra.
- "label" deve essere ESATTAMENTE uno tra:
  integra_con | compete_con | fornito_da | parte_di | usa | prezzo_di |
  alternativa_a | richiede | esempio_di | correlato_a
- Usa "correlato_a" SOLO come fallback se nessun'altra label è appropriata.
- Niente self-loops (src == dst).

OUTPUT — JSON valido, SOLO JSON:
{
  "entities": [
    {"name": "...", "type": "..."},
    ...
  ],
  "relations": [
    {"src": "...", "dst": "...", "label": "..."},
    ...
  ]
}

Massimo 10 entità e 8 relazioni per chunk. Se il chunk è vuoto/insignificante,
ritorna {"entities": [], "relations": []}.
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


def _validate_type(t: str) -> EntityType:
    t = (t or "").strip().lower()
    allowed = ("person", "org", "product", "concept", "location", "other")
    return t if t in allowed else "other"  # type: ignore[return-value]


def _validate_label(label: str) -> str:
    lbl = (label or "").strip().lower().replace(" ", "_")
    return lbl if lbl in ALLOWED_RELATION_LABELS else "correlato_a"


def extract_from_chunk(chunk_text: str, model: str | None = None) -> ExtractionResult:
    """Estrae entità + relazioni da un singolo chunk.

    Args:
        chunk_text: testo del chunk (max ~2000 chars; oltre viene tagliato).
        model: override del modello extractor.

    Returns:
        ExtractionResult (anche vuoto se il chunk è insignificante o errore).
    """
    text = (chunk_text or "").strip()
    if not text:
        return ExtractionResult(raw="")
    if len(text) > 2000:
        text = text[:2000]

    used_model = model or GRAPH.extractor_model

    try:
        resp = ollama.chat(
            model=used_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"CHUNK:\n{text}\n\nEstrai in JSON."},
            ],
            format="json",
            options={
                "temperature": 0.1,
                "num_predict": 600,
            },
            keep_alive=LLM.ollama_keep_alive,
            stream=False,
        )
        raw = (resp.get("message", {}).get("content") or "").strip()
    except Exception as e:
        print(f"[graph_extractor] LLM error: {e}")
        return ExtractionResult(raw=f"ERROR: {e}")

    parsed = _parse_json_loose(raw)
    raw_entities = parsed.get("entities") or []
    raw_relations = parsed.get("relations") or []

    entities: list[ExtractedEntity] = []
    seen_names: set[str] = set()
    for item in raw_entities[: GRAPH.max_entities_per_chunk]:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        if not name or len(name) < GRAPH.min_entity_len:
            continue
        # dedup case-insensitive
        key = name.lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        typ = _validate_type(item.get("type", "other"))
        entities.append(ExtractedEntity(name=name, type=typ))

    relations: list[ExtractedRelation] = []
    entity_keys = {e.name.lower() for e in entities}
    for item in raw_relations[: GRAPH.max_relations_per_chunk]:
        if not isinstance(item, dict):
            continue
        src = (item.get("src") or "").strip()
        dst = (item.get("dst") or "").strip()
        if not src or not dst or src.lower() == dst.lower():
            continue
        # Drop relations che fanno riferimento a entità non estratte
        if src.lower() not in entity_keys or dst.lower() not in entity_keys:
            continue
        label = _validate_label(item.get("label", "correlato_a"))
        relations.append(ExtractedRelation(src=src, dst=dst, label=label))

    return ExtractionResult(entities=entities, relations=relations, raw=raw)
