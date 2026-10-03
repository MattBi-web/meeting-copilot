"""Query rewriter: pulisce la trascrizione rumorosa in query semantica.

Le ultime righe di trascrizione sono spesso piene di:
- filler ("uhm", "cioè", "allora", "ehm")
- esitazioni ("vorrei sapere... mh... cioè")
- ripetizioni
- errori di Whisper (parole troncate)

Un embedding di questo testo è subottimale. Passandolo a un mini-LLM con
istruzione di "estrarre la query informativa", otteniamo una stringa molto
più "embedding-friendly".

Esempio:
- Input:  "uhm sì allora io vorrei sapere ehm cioè quanto costa la versione base
           perché ho sentito che è... mh"
- Output: "prezzo versione base"

Costo: ~200ms su llama3.2:3b (già in RAM via warmup).
Cache: opzionale, sulla stessa stringa input.
"""
from __future__ import annotations

import re
from functools import lru_cache

try:
    import ollama  # noqa: F401
except ImportError:
    ollama = None  # type: ignore[assignment]

from ..config import LLM, PROVIDERS, RAG


_SYSTEM_PROMPT = """Sei un estrattore di query per retrieval RAG.
Ricevi una trascrizione automatica (rumorosa) di una videochiamata.
Devi estrarre la query informativa MANTENENDO TUTTI gli elementi semanticamente
rilevanti per il retrieval.

═══ REGOLE STRINGENTI ═══

1. **PRESERVA NOMI PROPRI**: ProductA, ProductB, Producta, Acme Consulting, ContactCenterSuite,
   Twilio, Telnyx, Pipecat, Redis, FreePBX, OpenAI, Anthropic — TUTTI mantenuti.
   Anche se trascritti male (es. "Genesis" invece di "ContactCenterSuite"), MANTIENI il termine
   esatto dell'input — il sistema RAG saprà cercare entrambe le varianti.

2. **PRESERVA VERBI DI INTENT**: integrare, costare, durare, gestire, funzionare,
   confrontare, scalare, supportare — sono CRITICI per il senso della query.
   "integrare X con Y" ≠ "X e Y".

3. **PRESERVA NUMERI/DATI**: 8-12 settimane, latenza ms, 100+ agenti — mantieni
   le quantità e unità.

4. **RIMUOVI SOLO**: filler ("uhm", "cioè", "allora", "vabbè"), esitazioni
   ("vorrei sapere", "mi chiedevo", "ti volevo chiedere"), ripetizioni.

5. **LINGUA**: stessa dell'input.

6. **OUTPUT**: SOLO la query (5-15 parole tipiche), niente preamboli/quote.

7. **NOISE = ""**: Se è chiacchiera (saluti, "ok", "perfetto"), restituisci stringa vuota.

═══ ESEMPI ═══

Input: "uhm sì allora vorrei sapere quanto costa la versione base"
Output: prezzo versione base

Input: "ti chiedevo se ProductA si può integrare con ContactCenterSuite o TelephonyVendorX"
Output: integrazione ProductA con ContactCenterSuite e TelephonyVendorX

Input: "ma quanto tempo ci vuole per andare in produzione?"
Output: tempo per andare in produzione

Input: "che differenza c'è tra ProductA e ProductB?"
Output: differenza tra ProductA e ProductB

Input: "come gestite il GDPR sui dati delle call?"
Output: gestione GDPR dati call

Input: "ciao buongiorno come va"
Output:"""


@lru_cache(maxsize=128)
def _cached_rewrite(text: str, model: str) -> str:
    """Cache lru: stessa stringa input → stessa query rewritten."""
    if not text.strip():
        return ""
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Input: {text}\nOutput:"},
            ],
            options={
                "temperature": 0.1,    # deterministico
                "num_predict": 40,     # query breve
                "stop": ["\n", "Input:"],
            },
            keep_alive=LLM.ollama_keep_alive,
            stream=False,
        )
        rewritten = (resp.get("message", {}).get("content") or "").strip()
        # Cleanup: rimuovi quote, "Output:" residue, punteggiatura iniziale
        rewritten = re.sub(r'^(?:output:|"|\')', "", rewritten, flags=re.IGNORECASE).strip()
        rewritten = rewritten.strip('"\' .')
        return rewritten
    except Exception as e:
        print(f"[query_rewriter] errore: {e}, fallback su input raw")
        return text


def rewrite_query(text: str) -> str:
    """Esegue il rewriting se abilitato, altrimenti ritorna il testo originale.

    Args:
        text: stringa concatenata delle ultime righe trascritte.

    Returns:
        Query pulita (potrebbe essere stringa vuota se è solo chiacchiera).
    """
    if not RAG.query_rewriting_enabled:
        return text
    # FIX latency: il query_rewriter chiama ollama.chat direttamente
    # (bypassa provider abstraction). In cloud mode questo o fallisce
    # con TCP timeout o ricarica un modello local inutilmente, aggiungendo
    # 500-2000ms per ciclo. Il composer cloud (Haiku/Sonnet) gestisce
    # ottimamente query rumorose, quindi skippiamo proprio il rewriter.
    if PROVIDERS.llm != "local":
        return text
    if not text or not text.strip():
        return ""
    # Cap input length per evitare prompt mostruosi
    if len(text) > 800:
        text = text[-800:]
    rewritten = _cached_rewrite(text.strip(), RAG.query_rewriting_model)
    # Se l'LLM ha restituito vuoto = "noise" → ritorna stringa vuota
    # (il retriever si comporta di conseguenza)
    return rewritten


# ──────────────────────────────────────────────────────────────
# HyDE (Hypothetical Document Embeddings) — Fase 9.9
# ──────────────────────────────────────────────────────────────
_HYDE_SYSTEM_PROMPT = """Sei un esperto del dominio. Dato un quesito o intenzione informativa,
scrivi una breve risposta IPOTETICA come se la sapessi (3-4 frasi, max 80 parole).

REGOLE:
- Lingua: stessa del quesito (italiano o inglese).
- Niente preamboli ("Ecco la risposta..."), niente "Non ho informazioni": INVENTA un contenuto plausibile.
- Usa termini tecnici specifici, numeri verosimili, vocabolario del dominio.
- Lo scopo: produrre testo SIMILE a come sarebbero scritti i documenti di riferimento.
- NON serve essere accurato, serve "matchare" lessico/concetti dei documenti reali.

ESEMPIO:
Quesito: "prezzo voicebot"
Risposta ipotetica: "Il prezzo del voicebot parte da 290 euro al mese per la versione
base con 1000 ore di chiamate incluse. La versione enterprise offre chiamate illimitate
e SLA dedicato a partire da 1200 euro/mese. Il pricing è scalabile in base al numero
di utenti e all'integrazione con CRM esistenti."
"""


@lru_cache(maxsize=64)
def _cached_hyde(query: str, model: str) -> str:
    if not query.strip():
        return ""
    try:
        import ollama
        from ..config import LLM as _LLM
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Quesito: {query}\nRisposta ipotetica:"},
            ],
            options={
                "temperature": 0.7,
                "num_predict": 120,
            },
            keep_alive=_LLM.ollama_keep_alive,
            stream=False,
        )
        return (resp.get("message", {}).get("content") or "").strip()
    except Exception as e:
        print(f"[hyde] errore: {e}, fallback su query raw")
        return query


def expand_query_hyde(query: str) -> str:
    """Hypothetical Document Embeddings.

    Riceve una query (idealmente già rewritten), produce una "risposta ipotetica"
    plausibile, e ritorna la concatenazione `query + risposta`. L'embedding di
    questa concatenazione tende a "matchare" meglio i documenti reali nel KB
    (perché il lessico è già del dominio).

    Costo: 1 LLM call extra (~500ms su llama3.2:3b).
    """
    if not RAG.hyde_enabled:
        return query
    if not query.strip():
        return query
    hyde_doc = _cached_hyde(query.strip(), RAG.hyde_model)
    if not hyde_doc:
        return query
    # Concatena query + ipotetica: l'embedding peserà entrambe le componenti
    return f"{query}\n\n{hyde_doc}"
