"""Composer agent: genera la risposta narrativa da mostrare nell'overlay.

Modello: qwen2.5:14b-instruct (default, ~2-3s su M5).
Input: categoria + query + chunks recuperati dalla KB + sezioni attive.
Output: markdown strutturato (3-5 righe), STREAMING token-by-token.

Per ogni categoria usa un system prompt diverso per modulare il tono:
- question         → risposta diretta + 1 follow-up opzionale
- objection        → ack della preoccupazione + risposta data-driven
- pitch_opportunity → 1-2 frasi punchy per "agganciare" il discorso
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Callable

# ollama opzionale (solo PROVIDERS.llm=local). Cloud mode → bundle escluso.
try:
    import ollama  # noqa: F401
except ImportError:
    ollama = None  # type: ignore[assignment]

from ..config import LLM
from ..kb.retriever import SearchHit


_RULES_HIGH = """
═══ REGOLE — GROUNDING HIGH (rispondi SOLO dalla KB) ═══

I chunks forniti contengono materiale rilevante alla domanda.
La tua risposta DEVE essere costruita SOLO sulla base di questi chunks.

1. **OGNI claim deve essere nei chunks**: numeri, nomi, integrazioni, durate.
   NON inferire, NON estrapolare, NON inventare.
2. **CITAZIONI inline OBBLIGATORIE** alla fine di ogni frase rilevante:
   formato `(fonte: nome_file.md)`. Se non puoi citare → non scriverlo.
3. **NOMI PROPRI** esattamente come scritti nei chunks.
4. **LINGUA**: stessa della query.

═══════════════════════════════════════════
"""

_RULES_MEDIUM = """
═══ REGOLE — GROUNDING MEDIUM (KB parziale + integrazione LLM) ═══

I chunks contengono info parziale sulla domanda. Puoi integrare con
conoscenza generale, MA DICHIARANDOLO ESPLICITAMENTE.

Struttura risposta:
1. **Inizia con quello che è nella KB** (con citazione obbligatoria).
2. **Poi DICHIARA**: "Integro con conoscenza generale del settore:"
3. Aggiungi 1-2 frasi di knowledge esterna pertinenti.
4. **NON inventare numeri specifici** (%, prezzi, durate esatte) se non sono nella KB.
5. **LINGUA**: stessa della query.

Esempio struttura:
"Nella KB attiva trovo che [X dato] (fonte: file.md).
Integro con conoscenza generale del settore: [aggiunta contestuale generica, no numeri inventati]."

═══════════════════════════════════════════
"""

_RULES_LOW = """
═══ REGOLE — GROUNDING LOW (KB insufficiente → usa LLM knowledge piena) ═══

I chunks NON contengono info specifica per questa domanda. RISPONDI COMUNQUE
con la tua knowledge LLM di settore, in modo sostanzioso e utile. Il consulente
ti sta chiamando in real-time durante una call — gli serve una risposta usabile,
non un disclaimer lungo.

Struttura risposta:
1. **Disclaimer compatto SOLO inline come prefix**: `📎 (non da KB)` all'inizio
   della risposta. UNA volta sola. Niente warning grandi né emoji ⚠️.
2. **Risposta sostanziosa**: 2-3 frasi piene di contenuto tecnico/di settore.
   Usa la tua knowledge: vendor noti (Twilio, ContactCenterSuite, AWS, OpenAI, ecc),
   pattern di integrazione standard, paragoni con prodotti competitor.
3. **DIVIETO HARD**: NON inventare numeri/prezzi/timeline specifici sul prodotto
   discusso (es. ProductA/ProductB). Per esterni (ContactCenterSuite SLA standard,
   prezzi cloud noti) puoi citare ordini di grandezza tipici del settore.
4. **NIENTE filler tipo "prendo nota e verifico post-call"** — è inutile.
   Dai info concrete; il consulente sa già che approfondirà dopo.
5. **LINGUA**: stessa della query.

ESEMPIO BUONO:
"📎 (non da KB) ContactCenterSuite è una piattaforma CCaaS enterprise; integrazione tipica
via SIP trunking o REST/WebSocket ContactCenterSuite API. ProductA ha architettura
provider-agnostic quindi un adapter custom è fattibile, simile a quello Twilio."

ESEMPIO CATTIVO (da evitare):
"⚠️ Nella KB attiva non trovo info su ContactCenterSuite. Posso darti contesto generale?
Knowledge generale: forse serve un adapter. Prendo nota e verifico post-call."

═══════════════════════════════════════════
"""


def _rules_for_tier(tier: str) -> str:
    """Sceglie le regole anti-hallucination in base al confidence tier RAG."""
    if tier == "high":
        return _RULES_HIGH
    if tier == "medium":
        return _RULES_MEDIUM
    return _RULES_LOW  # low / unknown


# Backward compat per moduli legacy che importano _COMMON_PROMPT_RULES
_COMMON_PROMPT_RULES = _RULES_MEDIUM

_GLOBAL_PREAMBLE = """═══ CONTESTO OPERATIVO — LEGGI PRIMA DI OGNI RISPOSTA ═══

Il tuo utente è un CONSULENTE IN CALL LIVE col cliente — sta parlando in
videochiamata ADESSO. Tu sei un overlay che gli mostra suggerimenti veloci:
lui ti legge di sfuggita mentre conversa, non può rispondere a niente che tu
gli scriva.

═══ REGOLE ASSOLUTE — viola ANCHE UNA SOLA = output inutilizzabile ═══

1. MAI fare domande retoriche al consulente. Vietate ESPRESSAMENTE:
   ❌ "Vuoi che approfondisca?"
   ❌ "Intendi X o Y?"
   ❌ "Hai bisogno di chiarimenti?"
   ❌ "Posso continuare?"
   ❌ "Ti serve altro?"
   Sono inutili: lui non può risponderti, sta parlando col cliente.

2. SEI UN GHOST WRITER, non un chatbot. Il tuo output deve essere DIRETTAMENTE
   ripetibile dal consulente al cliente, oppure un consiglio operativo SECCO.
   Formati validi:
   ✓ "Digli: <frase pronta da dire>"
   ✓ "Proponi: <frase pronta>"
   ✓ "Dato chiave: <numero/nome>"
   ✓ Risposta in stile parlato, diretta al cliente.
   ✓ "Se chiede X → rispondi: <Y>"

3. MAI meta-commentare. Se la query è ambigua: dai 2 ipotesi azionabili
   brevi, NON chiedere quale intendeva.

4. Se il consulente sta GIÀ parlando bene da solo (transcript mostra che ha
   detto qualcosa di valido), NON correggere — dai materiale di rinforzo
   (numeri, esempi, frase di chiusura).

═══════════════════════════════════════════
"""


_CATEGORY_INTRO = {
    "question": """L'interlocutore (cliente) ha appena fatto una domanda. Il consulente
deve rispondere a voce. Produci frasi PRONTE PER LUI DA DIRE AL CLIENTE.""",
    "objection": """L'interlocutore (cliente) ha sollevato un'OBIEZIONE o dubbio.
Produci una contro-argomentazione che il consulente possa dire a voce.""",
    "pitch_opportunity": """L'interlocutore (cliente) ha menzionato un topic dove il
consulente ha materiale rilevante. Suggerisci un aggancio testuale CHE LUI PUÒ DIRE.""",
}

_CATEGORY_FORMAT = {
    "question": """
FORMATO (ghost writer mode):
- 2-3 frasi totali, max 50 parole, PRONTE DA DIRE AL CLIENTE
- Stile parlato, prima persona singolare/plurale (es. "Noi gestiamo...", "Possiamo...")
- NIENTE preamboli ("Certo", "Ecco")
- NIENTE domande retoriche al consulente (vedi preambolo)
- **bold** SOLO per 1 dato chiave
- Finisci con punto fermo
""",
    "objection": """
FORMATO (contro-argomentazione da dire):
- 3 frasi PRONTE DA DIRE: (1) ack ≤8 parole "Capisco il punto, ...", (2) contro-fatto,
  (3) chiusura propositiva ("Possiamo mostrarvi X subito.")
- Max 55 parole, stile parlato prima persona
- Niente domande retoriche al consulente
""",
    "pitch_opportunity": """
FORMATO (aggancio pitch da dire):
- 2 frasi PRONTE DA DIRE, max 40 parole
- Pattern: "Sul tema X — [dato concreto]. Possiamo fare Y."
- 1 dato citato se tier high/medium
- Stile parlato, niente domande retoriche
""",
}


_LANG_NAMES = {
    "it": "italiano",
    "en": "inglese",
    "es": "spagnolo",
    "fr": "francese",
    "de": "tedesco",
    "pt": "portoghese",
}


def _build_language_directive() -> str:
    """Costruisce direttiva di lingua output basata su LLM.suggestion_language."""
    target = getattr(LLM, "suggestion_language", "auto")
    if target == "auto":
        return (
            "\n═══ LINGUA OUTPUT ═══\n"
            "Rispondi nella STESSA lingua dell'interlocutore (mirror della query).\n"
        )
    name = _LANG_NAMES.get(target, target)
    return (
        f"\n═══ LINGUA OUTPUT (FORCED) ═══\n"
        f"Rispondi SEMPRE in {name}, INDIPENDENTEMENTE dalla lingua dell'interlocutore.\n"
        f"Il consulente vuole leggere i suggerimenti in {name} per processarli più veloce.\n"
    )


def _build_system_prompt(category: str, tier: str) -> str:
    """Combina preambolo globale (ruolo + regole assolute) + intro categoria
    + regole tier + formato + direttiva lingua."""
    intro = _CATEGORY_INTRO.get(category, _CATEGORY_INTRO["question"])
    rules = _rules_for_tier(tier)
    fmt = _CATEGORY_FORMAT.get(category, _CATEGORY_FORMAT["question"])
    lang = _build_language_directive()
    return _GLOBAL_PREAMBLE + "\n" + intro + "\n" + rules + "\n" + fmt + "\n" + lang


# Backward compat: alcuni test legacy importano _SYSTEM_PROMPTS
_SYSTEM_PROMPTS = {
    cat: _build_system_prompt(cat, "medium")
    for cat in ("question", "objection", "pitch_opportunity")
}


_USER_TEMPLATE = """**Categoria intent:** {category}
**Query estratta dall'analyzer:** {query}

**Chunks rilevanti dalla knowledge base attiva:**
{chunks}

**Ultime righe della conversazione (per contesto, NON ripeterle):**
{transcript}

Genera ora il suggerimento in markdown."""


def _format_chunks(hits: list[SearchHit]) -> str:
    if not hits:
        return "(nessun chunk rilevante)"
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(
            f"---\n[{i}] Sezione: {h.section} · Fonte: {h.source} · Score: {h.score:.2f}\n"
            f"{h.text}\n"
        )
    return "\n".join(parts)


def _pick_model_for_category(default_model: str, category: str) -> str:
    """Latency #5: composer ibrido per categoria.

    question + pitch_opportunity → Haiku 4.5 (TTFT ~500ms, full ~2s).
    objection → Sonnet 4.6 (qualità diplomazia + ragionamento sottile).

    Quando PROVIDERS.llm == "cloud": il cloud provider rimappa lo string
    "haiku"/"sonnet" al modello effettivo (vedi CLOUD.analyzer/composer_remap).
    Quando local: ignora la decisione e usa default_model (qwen2.5:7b).
    """
    try:
        from ..config import PROVIDERS
        if PROVIDERS.llm != "cloud":
            return default_model
    except Exception:
        return default_model
    if category == "objection":
        return "claude-sonnet-4-6"   # qualità top per gestire obiezione
    # question, pitch_opportunity, default → Haiku veloce
    return "claude-haiku-4-5"


class Composer:
    """Wrapper riusabile per il composer agent."""

    def __init__(self, model: str | None = None):
        self.model = model or LLM.composer_model

    def compose_stream(
        self,
        category: str,
        query: str,
        chunks: list[SearchHit],
        transcript_lines: list[str],
        on_token: Callable[[str], None] | None = None,
        grounding_tier: str = "medium",
    ) -> str:
        """Stream del testo. Chiama `on_token(piece)` per ogni delta.
        Ritorna il testo finale completo.

        Args:
            grounding_tier (Fase 13): "high" | "medium" | "low".
              Determina il system prompt e quindi il comportamento del composer
              rispetto al grounding sulla KB.
        """
        # Fase 13: usa prompt diverso in base al confidence tier RAG
        system_prompt = _build_system_prompt(category, grounding_tier)
        # Prepend session prompt (meta-istruzioni utente) se presente
        try:
            from ..session_prompt import prepend_to_system
            system_prompt = prepend_to_system(system_prompt)
        except Exception:
            pass
        user_msg = _USER_TEMPLATE.format(
            category=category,
            query=query,
            chunks=_format_chunks(chunks),
            transcript="\n".join(transcript_lines[-LLM.composer_transcript_lines:]),
        )
        # Prepend rolling summary (Context L3) come background
        try:
            from ..meeting_summary import prepend_to_user_msg
            user_msg = prepend_to_user_msg(user_msg)
        except Exception:
            pass

        # Usa provider abstraction (supporta local/remote/cloud).
        # Quando PROVIDERS.llm=cloud → Claude Sonnet 4.6 streaming
        # invece di Ollama local qwen2.5:7b (5-10s su CPU).
        accumulator: list[str] = []
        try:
            from ..providers import get_llm_provider
            from ..providers.base import LLMMessage
            provider = get_llm_provider()

            def _on_token(piece: str) -> None:
                accumulator.append(piece)
                if on_token:
                    on_token(piece)

            # Latency #5: composer ibrido — Haiku per question/pitch, Sonnet per objection
            effective_model = _pick_model_for_category(self.model, category)
            if effective_model != self.model:
                print(f"[composer] category={category} → model={effective_model}")
            provider.complete_stream(
                model=effective_model,
                messages=[
                    LLMMessage("system", system_prompt),
                    LLMMessage("user", user_msg),
                ],
                temperature=0.4,
                max_tokens=LLM.composer_max_tokens,
                on_token=_on_token,
            )
        except Exception as e:
            err_str = str(e)
            if "404" in err_str or "not found" in err_str.lower() or "manifest" in err_str.lower():
                err_msg = (
                    f"\n\n⚠️ **Errore composer**: modello `{self.model}` non disponibile.\n\n"
                    f"Fix: cambia composer model (Settings → LLM) o aggiorna Cloud remap.\n\n"
                    f"Dettaglio: {err_str[:200]}"
                )
            else:
                err_msg = f"\n\n⚠️ Errore composer: {e}"
            accumulator.append(err_msg)
            if on_token:
                on_token(err_msg)

        return "".join(accumulator)

    def compose(
        self,
        category: str,
        query: str,
        chunks: list[SearchHit],
        transcript_lines: list[str],
    ) -> str:
        """Versione sincrona senza streaming (rare cases)."""
        return self.compose_stream(category, query, chunks, transcript_lines, on_token=None)
