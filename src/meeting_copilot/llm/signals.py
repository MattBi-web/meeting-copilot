"""L0 signals: detectors rule-based veloci (1-5 ms) sulle ultime righe.

Ogni detector ritorna un Signal (score 0..1 + spiegazione).
Signals possibili:
- question        → pattern di domanda (sintassi, parole interrogative)
- objection       → markers di dubbio/contestazione
- named_entity    → menzione di competitor/prodotto/dato tecnico/prezzo
- info_request    → richiesta esplicita di info ("mi dici", "fammi sapere")
- pause_after     → segmenti corti finali = chi parla ha fatto una pausa interrogativa

I detector funzionano su italiano + inglese.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Signal:
    name: str
    score: float        # 0..1
    explanation: str    # breve, per debug/log


# ──────────────────────────────────────────────────────────────
# Question detector
# ──────────────────────────────────────────────────────────────
# Italian + English question markers
_QUESTION_MARKERS_IT = (
    r"\b(?:come|cosa|perch[éè]|quando|dove|chi|quale|quanto|qual[eèé]|"
    r"posso|potrei|potremmo|si pu[oò]|"
    r"avete|hai|hanno|ha\b|ho\b|fai|fate|fanno|"
    r"vorrei sapere|mi diresti|mi puoi dire|sai|sapete|"
    r"siete in grado|riuscireste|riesci|"
    r"funziona|com'?è|che ne pensa)"
)
_QUESTION_MARKERS_EN = (
    r"\b(?:what|how|why|when|where|who|which|"
    r"could you|can you|would you|do you|does it|"
    r"is there|are there|tell me|let me know|"
    r"i'?d like to know|how does|how do)"
)


def detect_question(text: str) -> Signal:
    """Score 0..1 per probabilità che il testo contenga una domanda."""
    if not text:
        return Signal("question", 0.0, "empty")

    text_lower = text.lower().strip()
    base_score = 0.0
    reasons = []

    # Bonus se finisce con ?
    if text_lower.rstrip(".… ").endswith("?"):
        base_score = 0.85
        reasons.append("'?' finale")
    # Markers interrogativi
    if re.search(_QUESTION_MARKERS_IT, text_lower):
        base_score = max(base_score, 0.7)
        reasons.append("marker IT")
    if re.search(_QUESTION_MARKERS_EN, text_lower):
        base_score = max(base_score, 0.7)
        reasons.append("marker EN")

    # Pattern "mi spieghi" / "spiegami" → richiesta esplicita
    if re.search(r"\b(?:mi spieghi|spiegami|explain to me|can you explain)", text_lower):
        base_score = max(base_score, 0.9)
        reasons.append("richiesta spiegazione")

    return Signal("question", base_score, ", ".join(reasons) or "no_match")


# ──────────────────────────────────────────────────────────────
# Objection detector
# ──────────────────────────────────────────────────────────────
_OBJECTION_MARKERS_IT = (
    r"\b(?:per[oò]|tuttavia|ma\s|non sono sicur|ho dei dubbi|"
    r"non mi convince|sembra costos|troppo car|caro|"
    r"non funzionerebbe|impossibile|non vedo come|"
    r"problema|ostacolo|rischio|preoccupazione)"
)
_OBJECTION_MARKERS_EN = (
    r"\b(?:however|but\s|not sure|i'?m not sure|"
    r"too expensive|too costly|seems pricey|"
    r"doesn'?t work|won'?t work|problem|risk|concern|worry|hesitation)"
)


def detect_objection(text: str) -> Signal:
    if not text:
        return Signal("objection", 0.0, "empty")
    t = text.lower()
    reasons = []
    score = 0.0
    if re.search(_OBJECTION_MARKERS_IT, t):
        score = max(score, 0.7)
        reasons.append("marker IT")
    if re.search(_OBJECTION_MARKERS_EN, t):
        score = max(score, 0.7)
        reasons.append("marker EN")
    return Signal("objection", score, ", ".join(reasons) or "no_match")


# ──────────────────────────────────────────────────────────────
# Info request detector ("vorrei sapere", "puoi mandarmi", "fammi sapere")
# ──────────────────────────────────────────────────────────────
_INFO_REQUEST_MARKERS = (
    r"\b(?:vorrei sapere|fammi sapere|mi diresti|mi puoi mandare|"
    r"avresti info|cerco info|mi servirebbe|mi farebbe piacere sapere|"
    r"i need to know|i'?d like to know|can you send|let me know|"
    r"i'?m looking for)"
)


def detect_info_request(text: str) -> Signal:
    if not text:
        return Signal("info_request", 0.0, "empty")
    if re.search(_INFO_REQUEST_MARKERS, text.lower()):
        return Signal("info_request", 0.75, "marker matched")
    return Signal("info_request", 0.0, "no_match")


# ──────────────────────────────────────────────────────────────
# Named entity detector (lightweight, heuristic)
# Cerca: nomi propri/aziende/prezzi/numeri tecnici significativi.
# ──────────────────────────────────────────────────────────────
# Capitalized words seq (>=2 char), escludendo inizio frase
_CAP_SEQ = re.compile(r"(?<![\.\?!]\s)\b[A-Z][a-zA-Z0-9]{2,}\b(?:\s[A-Z][a-zA-Z0-9]+)*")

# Prezzi (€/$ con cifre, o "N euro/dollari")
_PRICE_PATTERN = re.compile(
    r"(?:[€$]\s?\d+(?:[.,]\d+)?(?:\s*[kKmM])?|\b\d+(?:[.,]\d+)?\s*(?:euro|euri|dollar(?:i|s)|usd|eur)\b)",
    re.IGNORECASE,
)

# Misure tecniche (ms, GB, %, etc.)
_TECH_NUMBER_PATTERN = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:ms|millisecond[i]?|secondi?|sec|"
    r"giga|gb|tb|terabyte|mb|kb|"
    r"percento|%|"
    r"ore|minuti?|giorni?|"
    r"chiamate?|users?|utenti)\b",
    re.IGNORECASE,
)


def detect_named_entity(text: str) -> Signal:
    if not text:
        return Signal("named_entity", 0.0, "empty")

    matches: list[str] = []
    score = 0.0

    cap_matches = _CAP_SEQ.findall(text)
    # Filtra titoli generici comuni
    blacklist_caps = {
        "Ciao", "Salve", "Buongiorno", "Buonasera", "Hello", "Hi", "Ok", "Si", "No",
        "Allora", "Quindi", "Oggi", "Ieri", "Domani",
    }
    cap_matches = [c for c in cap_matches if c not in blacklist_caps]
    if cap_matches:
        score = max(score, 0.5)
        matches.append(f"caps={cap_matches[:3]}")

    if _PRICE_PATTERN.search(text):
        score = max(score, 0.75)
        matches.append("price")

    if _TECH_NUMBER_PATTERN.search(text):
        score = max(score, 0.65)
        matches.append("tech_num")

    return Signal("named_entity", score, ", ".join(matches) or "no_match")


# ──────────────────────────────────────────────────────────────
# Small talk detector (suppressore: alza la soglia di trigger)
# ──────────────────────────────────────────────────────────────
_SMALLTALK_MARKERS = (
    r"\b(?:ciao|salve|buongiorno|buonasera|buonanotte|"
    r"come stai|come va|tutto bene|piacere|grazie|prego|"
    r"hello|hi\s|hey\s|good morning|good afternoon|good evening|"
    r"how are you|how'?s it going|nice to meet|thank you|thanks|you'?re welcome|"
    r"^ok$|^okay$|^si$|^certo$|^ah$|^uhm$|^ehm$|^mm$|^mh$)"
)


def detect_smalltalk(text: str) -> Signal:
    if not text:
        return Signal("smalltalk", 0.0, "empty")
    if re.search(_SMALLTALK_MARKERS, text.lower().strip()):
        return Signal("smalltalk", 0.8, "small_talk marker")
    # Frase molto corta (< 5 parole) → probabilmente filler
    words = text.split()
    if len(words) <= 3:
        return Signal("smalltalk", 0.4, f"short_filler ({len(words)} words)")
    return Signal("smalltalk", 0.0, "no_match")


# ──────────────────────────────────────────────────────────────
# Compute all L0 signals on a window of recent lines
# ──────────────────────────────────────────────────────────────
@dataclass
class L0Result:
    signals: dict[str, Signal]
    salience_score: float       # combined "interesting?" 0..1
    suppression_score: float    # combined "boring/skip?" 0..1
    notes: list[str]


def compute_l0(recent_lines: list[str]) -> L0Result:
    """Esegue tutti i detector L0 su una finestra di righe recenti.

    Heuristic di aggregazione:
    - salience = max(question, objection, info_request, named_entity)
    - suppression = max(smalltalk * 0.7)  (smalltalk non blocca completamente)
    """
    if not recent_lines:
        return L0Result(
            signals={},
            salience_score=0.0,
            suppression_score=1.0,
            notes=["no_recent_lines"],
        )

    # Considera l'unione delle ultime 3 righe per i detector "richiesta info"
    joined = " ".join(recent_lines[-3:])
    last_line = recent_lines[-1]

    sigs = {
        "question": detect_question(last_line),
        "objection": detect_objection(joined),
        "info_request": detect_info_request(joined),
        "named_entity": detect_named_entity(joined),
        "smalltalk": detect_smalltalk(last_line),
    }

    salience = max(
        sigs["question"].score,
        sigs["objection"].score,
        sigs["info_request"].score,
        sigs["named_entity"].score,
    )
    suppression = sigs["smalltalk"].score * 0.7

    notes = [f"{k}={v.score:.2f}({v.explanation})" for k, v in sigs.items() if v.score > 0.1]

    return L0Result(
        signals=sigs,
        salience_score=salience,
        suppression_score=suppression,
        notes=notes,
    )
