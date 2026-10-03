"""Central configuration for Meeting Copilot."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# ----- .env (dev convenience) -----
# Carica <repo>/.env se esiste: le chiavi (Deepgram, Anthropic, ecc.)
# stanno lì invece che nell'export manuale del terminale. Env già settate
# nel processo NON vengono sovrascritte (override=False).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
except Exception:
    pass  # dotenv assente o file mancante: si usano le env del processo

# ----- Paths -----
# Delega a paths.py che gestisce sia dev mode che bundled mode (.app PyInstaller).
# In bundled mode i path scrivibili vanno in ~/Library/Application Support/MeetingCopilot/,
# in dev mode restano nel repo (compatibilità storica).
#
# Logica esplicita (no silent fallback): se paths.py manca o se siamo bundled,
# preferiamo sempre la dir OS-standard. Solo in dev pulito, fallback al repo.
_IS_BUNDLED = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

if _IS_BUNDLED:
    # Bundled mode: SEMPRE Library/Application Support (Mac) o AppData (Win).
    # Non passiamo da paths.py per evitare qualsiasi fallback indesiderato.
    if sys.platform == "darwin":
        PROJECT_ROOT = Path.home() / "Library" / "Application Support" / "MeetingCopilot"
    elif sys.platform == "win32":
        _appdata = os.environ.get("APPDATA")
        PROJECT_ROOT = Path(_appdata) / "MeetingCopilot" if _appdata else Path.home() / "AppData" / "Roaming" / "MeetingCopilot"
    else:
        PROJECT_ROOT = Path.home() / ".local" / "share" / "MeetingCopilot"
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[config] bundled mode: PROJECT_ROOT={PROJECT_ROOT}")
else:
    # Dev mode (repo): paths storici
    try:
        from .paths import user_data_dir as _udd, transcripts_dir as _tdir
        PROJECT_ROOT = _udd()
        TRANSCRIPTS_DIR = _tdir()
    except Exception as e:
        print(f"[config] paths.py import failed: {e!r} — using repo root fallback")
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
        TRANSCRIPTS_DIR.mkdir(exist_ok=True)


# ----- Audio -----
@dataclass(frozen=True)
class AudioConfig:
    # Nome dell'Aggregate Device creato in Fase 0
    device_name: str = "MeetingCapture"

    # Whisper richiede 16 kHz mono. Lo otteniamo via downmix + resample.
    sample_rate: int = 16_000

    # Channels letti dal device. MeetingCapture su MacBook Air = 3 canali
    # totali (BlackHole 2ch stereo + Microfono integrato mono).
    # Li miscelermo a mono via mean. Se cambi setup audio, aggiorna qui.
    input_channels: int = 3

    # Frame size del callback (in samples a sample_rate=16k):
    # 512 samples = 32 ms — match con silero-vad streaming.
    block_size: int = 512


# ----- VAD -----
@dataclass(frozen=True)
class VADConfig:
    # Soglia probabilità "è voce" sopra cui consideriamo voce attiva.
    speech_threshold: float = 0.5

    # Quanto silenzio (ms) considerare come fine-frase → trigger flush a STT.
    # Fase 9.1.A — abbassato a 300ms per percezione streaming più reattiva.
    min_silence_ms: int = 300

    # Lunghezza massima di un segmento prima di forzare flush (ms).
    # Fase 9.1.A — TUNING: 4000ms invece di 10000.
    # Tradeoff: più chunks = più overhead Whisper totale, ma utente vede
    # testo apparire molto più rapidamente (~4s percepiti invece di ~10s).
    # Whisper Turbo MLX su M5 gira a ratio 0.1-0.2x, quindi anche con chunks
    # corti la latenza per chunk è 0.5-1s. Net: feel streaming.
    max_segment_ms: int = 4_000

    # Lunghezza minima di un segmento per essere processato (ms).
    # Fase 9.1.A — abbassato a 700ms per matchare il nuovo max_segment 4s.
    # Sotto 700ms = solo filler, scartato.
    min_segment_ms: int = 700

    # Fase 9.1.A — Initial prompt: passa a Whisper le ultime N parole del
    # chunk precedente come "primer" di contesto. Riduce errori sui confini
    # di frase quando viene spezzata tra chunks. 8 parole è sweet spot:
    # abbastanza contesto, non così tante da indurre ripetizioni.
    initial_prompt_words: int = 8


# ----- STT (mlx-whisper / faster-whisper) -----
@dataclass(frozen=True)
class STTConfig:
    # Backend STT:
    #   "mlx"           → mlx-whisper (GPU/Neural Engine M-series, FAST) — DEFAULT
    #   "faster-whisper" → CTranslate2 (CPU only, fallback non-Apple-Silicon)
    #   "deepgram"      → Cloud STT (Deepgram REST batch). Qualità top,
    #                      latency 150-400ms, zero CPU sul Mac.
    #                      Richiede env DEEPGRAM_API_KEY. ~$0.0043/min.
    #                      Supporta keyword boosting via session_vocabulary.txt.
    #   "voxtral"       → STUB (Fase 9.1.B): Mistral Voxtral via mlx-voxtral.
    #                      ⚠️ License "Personal Use Only" — non commerciale.
    backend: str = "deepgram"

    # Modello Whisper. Su backend "mlx" usiamo modelli pre-convertiti in MLX
    # da Hugging Face (~mlx-community/whisper-*-mlx).
    # Su backend "faster-whisper" sono i nomi standard ("small", "medium",
    # "turbo", "large-v3").
    # Su mlx-whisper + M5: "large-v3-turbo" gira sotto i 0.1x ratio (10x realtime)
    # con qualità top — non c'è motivo di scendere a modelli più piccoli.
    model_size: str = "large-v3-turbo"

    # Device / compute_type: usati SOLO dal backend faster-whisper.
    # Su mlx-whisper il device è sempre GPU/ANE.
    device: str = "cpu"
    compute_type: str = "int8"

    # IMPORTANTE per le performance: lasciare language=None costa 1-1.5s
    # di overhead per OGNI segmento (Whisper rifa la detection).
    # Forzare la lingua elimina questo overhead.
    # Cambia a "en" via CLI: `python -m meeting_copilot.main --language en`
    #
    # Lingue supportate (Deepgram nova-3): "it", "en", "es", "fr", "de", "pt",
    # "nl", "pl", "ru", "ja", "ko", "zh", "ar", "tr", "uk", "vi", "hi", "id".
    # Su Whisper local supporta auto-detect lasciando None.
    language: str | None = "it"

    # Lascia che Whisper non aspetti — bias verso latenza bassa.
    beam_size: int = 1
    vad_filter: bool = False  # noi facciamo VAD a monte
    condition_on_previous_text: bool = False  # evita allucinazioni a catena

    # Thread CPU usati da CTranslate2. Su M5 con ~10 core,
    # 8 lascia ancora 2 core liberi e velocizza la trascrizione.
    # Se vuoi più margine per Ollama in Fase 4 → torna a 6.
    cpu_threads: int = 8

    # Latency #10: Deepgram WebSocket streaming mode.
    # Quando True + backend="deepgram": bypassa VAD locale e streama audio raw
    # via WebSocket persistente. Transcript appaiono live (interim ~50-150ms
    # post-parola, final ~300ms post-end-of-speech).
    # Quando False: usa REST batch (default, più stabile).
    streaming_enabled: bool = True  # opt-in finché non testato in produzione

    # Mic gain boost in dB. 0 = nessun boost, 6 = 2x, 12 = 4x.
    # Utile quando l'audio arriva debole (es. dall'altro PC via mic ambientale).
    # ⚠️ Amplifica anche rumore di fondo. Sweet spot 6-9 dB.
    mic_gain_db: float = 9.0

    # Deepgram endpointing (solo streaming mode). ms di silenzio per chiudere
    # un'utterance. Più basso = cattura prima frammenti corti / audio debole.
    # Default Deepgram = 10ms in nova-3 streaming. 150 = pausa naturale parlata.
    deepgram_endpointing_ms: int = 150

    # Deepgram utterance_end_ms: timeout finale dopo last word interim.
    # Più basso = transcript "final" arriva prima ma può troncare frasi lunghe.
    deepgram_utterance_end_ms: int = 1000

    # ─── Multi-language (Phase Multilingual) ───
    # Modalità trascrizione:
    #   "single"  → usa STT.language (default, comportamento legacy)
    #   "auto"    → Deepgram detect_language=true (rileva una lingua per call)
    #   "multi"   → Deepgram language=multi (code-switching nel turno, nova-3)
    transcription_mode: str = "single"

    # Lingue supportate quando l'utente alterna 2 lingue nello stesso meeting
    # (rilevante solo con transcription_mode="multi"). Tuple comune:
    # ("it", "en"), ("en", "es"), ("fr", "en") ecc.
    multilingual_languages: tuple[str, ...] = ("it", "en")


# ----- LLM (Fase 4 + 9.4/9.5: analyzer + composer) -----
@dataclass(frozen=True)
class LLMConfig:
    # Analyzer: classifier veloce (200-400ms). Opzioni in ordine velocità → qualità:
    #   "llama3.2:3b"     → 2GB,  default storico, qualità OK IT/EN
    #   "phi3.5"          → 2GB,  Microsoft, ottimo classifier
    #   "phi4-mini"       → 2.5GB, Microsoft Phi-4 Q4_K_M, qualità top per la sua size
    #   "qwen2.5:3b"      → 2GB,  multilingue, qualità simile a llama
    analyzer_model: str = "llama3.2:3b"

    # Soglia minima di confidence dell'analyzer per triggerare il composer.
    # 0.5 = trigger anche su segnali moderati (più suggerimenti, qualche più rumore).
    # 0.6+ = più conservativo. 0.4 = aggressivo.
    analyzer_min_confidence: float = 0.5

    # Quante righe di trascrizione passare all'analyzer (le più recenti).
    # 10 = ~1.5 min di conversazione. Era 20 ma aggiungeva ~500ms TTFT
    # Haiku senza miglioramento concreto di classification quality.
    analyzer_transcript_lines: int = 10

    # Composer: LLM. Opzioni VERIFICATE disponibili su Ollama registry:
    #   "qwen2.5:7b-instruct"     → 4.5GB, DEFAULT, ottimo bilancio
    #   "qwen2.5:14b-instruct"    → 9GB, qualità top ma 2x più lento
    #   "llama3.1:8b-instruct"    → 5GB, alternativa
    #   "mistral:7b-instruct"     → 4GB, alternativa
    # NB: qwen3:8b NON è disponibile su Ollama (manifest non esiste a maggio 2026).
    #     Quando uscirà, aggiornare qui.
    composer_model: str = "qwen2.5:7b-instruct"

    # Quante righe di trascrizione passare al composer come contesto.
    # 12 = ~2 min conversazione. Era 20 ma cumulato con rolling summary
    # e session prompt gonfiava troppo il prompt → TTFT più alto.
    composer_transcript_lines: int = 12

    # Cap sul numero di token generati dal composer.
    # 280 = margin safety per 3 frasi punchy senza taglio mid-sentence
    # (con prompt aggressivo che chiede max 3 frasi). Il composer di solito
    # finisce a ~150-180 tokens; 280 è il bumper anti-truncation.
    composer_max_tokens: int = 280

    # Tieni i modelli LLM SEMPRE in RAM (no unload dopo idle).
    # -1 = mai scaricare (int). "5m" = 5 minuti (string con unit, default Ollama).
    # Su M5 con 16GB+ RAM va benissimo tenerli su; con 8GB metti "5m".
    ollama_keep_alive: int = -1

    # Auto-mode: ogni quanti secondi triggerare l'analyzer in auto-mode.
    # In manual-mode si triggera solo su hotkey.
    # Default 3s = reattività call-natural senza saturare cost cloud.
    auto_interval_s: float = 3.0

    # Fast-path trigger: pattern matching IMMEDIATO su nuove righe trascritte.
    # Quando una riga contiene un trigger pattern forte (es. "?", "spiegami",
    # "come funziona") → lancia composer subito bypassando analyzer LLM.
    # Risparmia ~1-2s di latency. Trade-off: alcune trigger spurie (es. domande
    # retoriche), ma il composer cloud è cheap.
    fast_path_enabled: bool = True

    # Score minimo di pattern signal per attivare fast-path.
    # 0.85 = solo veri "?" finale o "spiegami/explain" → no falsi positivi.
    # 0.70 era troppo permissivo: triggerava su frasi come "Adesso mi è venuta
    # in mente un'altra cosa" perché contengono "cosa" (marker IT).
    fast_path_min_score: float = 0.85

    # Cooldown tra fast-path trigger (sec). Evita re-trigger su pattern simili.
    fast_path_cooldown_s: float = 6.0

    # Live rolling summary (Context L3): ogni N sec rigenera un riassunto della
    # call corrente via mini LLM call. Viene iniettato nel composer come
    # "background". Permette al composer di "ricordare" call lunghe.
    # 0 = disabilitato. 300 = ogni 5 min.
    rolling_summary_interval_s: int = 300

    # Min righe trascritte prima di iniziare a generare il summary.
    rolling_summary_min_lines: int = 15

    # Modalità di default: "manual" o "auto". Toggleable runtime via overlay.
    default_mode: str = "manual"

    # ─── Lingua suggerimenti (output composer) ───
    # Quale lingua usa il composer per le risposte mostrate in overlay:
    #   "auto" → stessa della query (mirror della lingua dell'interlocutore)
    #   "it"   → SEMPRE in italiano
    #   "en"   → SEMPRE in inglese
    #   "es"   → SEMPRE in spagnolo
    #   "fr"   → SEMPRE in francese
    # Use case "it": utente italiano in call inglese, vuole leggere suggerimenti
    # in italiano per processarli più velocemente.
    suggestion_language: str = "auto"

    # Latency #8: Pipelining cicli. Quando True, l'orchestrator usa due thread:
    # uno per la prep (analyzer + retrieval), uno per il composer.
    # Vantaggio: mentre il composer del ciclo N streamea, il prep del ciclo N+1
    # può già partire. -30-50% perceived latency su sequenze trigger ravvicinati.
    # Quando False (default safe): comportamento sequenziale legacy.
    pipelining_enabled: bool = True


# ----- Watcher Agent (Fase 6: trigger intelligence) -----
@dataclass(frozen=True)
class WatcherConfig:
    # Pesi per lo scoring finale:
    # final = w0 * pattern_salience + w1 * kb_similarity - suppression
    w0_pattern: float = 0.4         # peso L0 (regex/pattern)
    w1_kb: float = 0.6              # peso L1 (KB similarity)

    # Soglie (tier mapping, score 0..1):
    threshold_indicator: float = 0.30   # sotto = silent
    threshold_hook: float = 0.55        # sotto = solo indicator
    threshold_auto: float = 0.75        # sotto = solo hook; sopra = auto-fire

    # Se True, chiama l'analyzer LLM come "ratifica" prima del composer
    # quando un hook entra in speculative. False = più veloce, accetta L0+L1.
    l2_ratify: bool = False

    # Default: il watcher è attivo all'avvio
    default_enabled: bool = True


# ----- Provider routing (Fase 8: ports & adapters) -----
@dataclass(frozen=True)
class ProvidersConfig:
    """Quale backend usare per ogni servizio. 'local' = on-device. 'remote' = HTTP server (Fase 12)."""
    stt: str = "local"
    embedding: str = "local"
    llm: str = "cloud"
    vector_store: str = "remote"
    rerank: str = "remote"
    sparse_store: str = "remote"


# ----- RAG (Fase 7+9: retrieval quality) -----
@dataclass(frozen=True)
class RAGConfig:
    # ── Embedding model (Fase 9.2) ──
    # Modelli noti disponibili via Ollama:
    #   "nomic-embed-text"     → 768-dim, multilingue OK (baseline, default storico)
    #   "bge-m3"               → 1024-dim, multilingue TOP, supporta dense+sparse+colbert
    #   "mxbai-embed-large"    → 1024-dim, English-focused
    # Default ora bge-m3 per qualità superiore in italiano.
    # IMPORTANTE: cambiando questo, devi re-indicizzare TUTTE le sezioni KB.
    embedding_model: str = "bge-m3"

    # ── Query rewriting ──
    query_rewriting_enabled: bool = True
    query_rewriting_model: str = "llama3.2:3b"

    # ── Hybrid retrieval (Fase 9.6) ──
    # Combina dense (embeddings) + sparse (BM25) con Reciprocal Rank Fusion.
    # +30% recall@10 confermato da letteratura production.
    hybrid_retrieval_enabled: bool = True
    # Pesi RRF: dense (semantica) vs sparse (parole esatte/named entities).
    # Default 70/30 da letteratura; tunato 85/15 (2026-05-21) per favorire
    # ranking semantico su KB tecnica acme_demo, dove la query
    # "WebSocket ProductA" semanticamente è ben matchata da bge-m3 ma
    # il BM25 disperdeva su file generici.
    rrf_dense_weight: float = 0.85
    rrf_sparse_weight: float = 0.15
    # RRF parameter "k" (di solito 60). Bilancia rank position vs raw scores.
    rrf_k: int = 60

    # ── Filtering & dedup ──
    min_score_threshold: float = 0.001    # più permissivo: meglio chunks "borderline" che niente
    dedup_similarity_threshold: float = 0.90
    initial_top_k: int = 15               # più candidati da rerankizare
    # final_top_k 3 (era 5/4): -100ms TTFT composer per ogni chunk in meno.
    # Su KB tecnica con dense weight 0.85, i top-3 contengono >90% dell'info utile.
    final_top_k: int = 3                  # chunks finali al composer

    # ── Reranker (Fase 7.3 + 9.3) ──
    # DISABILITATO definitivamente (2026-05-21): su VM CCX23 CPU AMD vanilla
    # sentence-transformers gira a ~900ms/pair indipendentemente dalla dimensione
    # del modello (bge-v2-m3 568M = 24s, mxbai-xsmall 33M = 22s su 24 pairs).
    # Test ha mostrato top-4 stessi con/senza rerank → il valore aggiunto qui
    # non giustifica la latency. Riabilita SOLO se: (a) si sposta su VM con GPU,
    # (b) si usa rerank cloud (Cohere/Anthropic), (c) si torna a rerank local
    # su Mac MPS quando RAM lo permette.
    rerank_enabled: bool = False
    # Opzioni:
    #   "mixedbread-ai/mxbai-rerank-xsmall-v1"  → 70M, ~150ms, default veloce
    #   "BAAI/bge-reranker-base"                → 278M, ~600ms, multilingue decente
    #   "BAAI/bge-reranker-v2-m3"               → 568M, 24s su VM CPU (NO!)
    #   "BAAI/bge-reranker-v2-gemma"            → 2.51GB, max qualità
    # Su M5 con MPS, bge-reranker-v2-m3 vale la spesa di qualità.
    rerank_model: str = "mixedbread-ai/mxbai-rerank-xsmall-v1"

    # ── HyDE: Hypothetical Document Embeddings (Fase 9.9) ──
    # Per query molto vaghe/ambigue, genera con un LLM una "risposta ipotetica"
    # e fa embedding di QUELLA invece della query raw. Migliora retrieval su
    # query brevi o concetti astratti.
    # Costo: 1 chiamata LLM extra (~500ms su llama3.2:3b).
    hyde_enabled: bool = False
    hyde_model: str = "llama3.2:3b"

    # ── Parser preferito (Fase 9.8) ──
    # "auto"        → MarkitDown se installato, altrimenti parser specifici
    # "markitdown"  → forza MarkitDown (Microsoft, layout-aware unificato)
    # "specific"    → forza i parser specifici (pypdf, python-docx, ecc.)
    parser_preference: str = "auto"

    # ── Vector store backend (Fase 9.7) ──
    # Controllato anche da PROVIDERS.vector_store, ma teniamo qui il sub-backend
    # quando provider="local":
    #   "chromadb"  → default storico, HNSW
    #   "lancedb"   → Arrow-based, 3-5x più veloce, supporta multi-vector
    # NB: switching richiede re-index di tutte le sezioni.
    local_vector_backend: str = "chromadb"

    # ── Fase 13 — Confidence-gated hybrid knowledge sourcing ──
    # Permette al composer di sapere se rispondere SOLO sulla KB (high),
    # ibrido KB+LLM (medium), o LLM-only con disclaimer (low).
    confidence_gating_enabled: bool = True

    # Soglie sui top hit scores (post-rerank, già normalizzati 0..1):
    #   max_score ≥ high   AND ≥2 chunks ≥ high  → tier HIGH (grounded strict)
    #   max_score ≥ medium                         → tier MEDIUM (grounded + LLM integra)
    #   max_score < medium                         → tier LOW (LLM-only con warning)
    grounding_high_threshold: float = 0.55
    grounding_medium_threshold: float = 0.30

    # ── Fase 14 — Corrective RAG (CRAG) ──
    # Quando attivo, dopo il retrieval un Evaluator LLM valuta i chunks.
    # Se "partial", esegue re-query con query expansion (sinonimi/parafrasi)
    # e merge dei risultati. Latency +200-800ms per call ma recall +30-50%.
    # OFF di default: aggiunge 1-3s al ciclo che è il colpo più caro percepito.
    # Riabilita per call analitiche dove ti serve recall massimo, non in real-time.
    crag_enabled: bool = False

    # Quante query variants tentare in re-retrieve (default 2, max 3).
    crag_max_variants: int = 2

    # ── Fase 16 — Agentic orchestration multi-step ──
    # Quando attivo, sostituisce il flusso lineare con un Planner che decide
    # quali step eseguire (retrieve_kb, retrieve_memory, query_expand, evaluate,
    # compose_grounded, compose_fallback) e in che ordine.
    # L'AgentOrchestrator esegue il plan stepwise con state machine e retry.
    # Latency +200-500ms per planning, ma adatta il flusso al tipo di query.
    # Opt-in: default OFF per non rompere il flusso legacy fino a test.
    agentic_mode_enabled: bool = False

    # Modello per il planner agent (deve produrre JSON strutturato).
    # Usa llama3.2:3b per velocità; switch a qwen2.5:7b per qualità superiore.
    agentic_planner_model: str = "llama3.2:3b"

    # Max step di un plan prima di forzare conclusione (anti-loop).
    agentic_max_steps: int = 6


# ----- Telemetry (Fase 10: adaptive learning) -----
@dataclass(frozen=True)
class TelemetryConfig:
    # Master switch. Tutto local, single-user, opt-in.
    enabled: bool = True

    # Quante decisioni recenti considerare per ricalcolo soglie.
    learning_window: int = 50

    # Soglia minima di dati per fidarsi del feedback (evita over-fitting).
    min_samples_to_learn: int = 20

    # Click rate target (soglia hook viene alzata se sotto, abbassata se sopra).
    click_rate_low: float = 0.15
    click_rate_high: float = 0.50

    # Quanto thumbs servono prima di toccare threshold_auto.
    min_thumbs_for_auto_adjust: int = 5

    # Delta per update soglie (graduale).
    threshold_delta: float = 0.02

    # Ogni quante decision (tier!=silent) ricalcolare automaticamente.
    recompute_every_n: int = 25



# ----- Session prompt (meta-instructions per call) -----
@dataclass(frozen=True)
class SessionConfig:
    """Session-level meta-prompt iniettato nel composer prima di ogni risposta.

    Use case: l'utente prima di una call può impostare istruzioni di contesto
    ("sono in una call OurCo↔Acme Consulting, KB attiva è acme_demo, rispondi
    technical-grade, segnala '[non da KB]' quando manca info, ecc").

    Il context_prompt viene PREPENDATO al system prompt del composer
    (sia local sia cloud sia cloud_burst), mantenendo intatte le regole
    di grounding (HIGH/MEDIUM/LOW) e anti-hallucination.

    Persistito su disco in PROJECT_ROOT/session_prompt.txt così sopravvive
    al restart del server. Editabile via dashboard /admin tab Start.
    """
    # Persist path (relativo a PROJECT_ROOT)
    persist_file: str = "session_prompt.txt"

    # Max char (cap per non gonfiare ogni call LLM)
    max_chars: int = 4000


# ----- Budget cap & cost tracking (Fase 23) -----
@dataclass(frozen=True)
class BudgetConfig:
    """Tracking spesa cloud (Anthropic/OpenAI/DeepSeek) + budget cap.

    Tutte le chiamate cloud (CloudLLM provider + cloud_burst) registrano
    token usage in SQLite. Se la spesa supera la soglia daily/monthly,
    le successive call cloud vengono BLOCCATE (errore esplicito) o degradate
    a fallback (se enabled).

    Indispensabile prima di distribuire ad altri o lasciare overlay
    aperto la notte: previene "carta vuotata accidentalmente".

    Pricing hardcoded per i modelli più usati. Aggiorna se cambiano.
    """
    enabled: bool = True

    # Cap giornaliero in USD. Quando superato → blocca.
    # 0 = nessun cap (sconsigliato in produzione).
    daily_cap_usd: float = 5.0

    # Cap mensile in USD (rolling 30 giorni).
    monthly_cap_usd: float = 50.0

    # Quando >= warning_threshold * cap → log warning ma non blocca
    warning_threshold: float = 0.80

    # Pricing per 1M token (input, output). Valori al 2026-05.
    # Aggiorna in pyproject o config se cambia.
    pricing_per_mtok: dict | None = None

    # Database SQLite location (relativo a PROJECT_ROOT)
    db_path: str = "budget.db"


# ----- Cloud LLM provider (Fase 22: provider cloud always-on) -----
@dataclass(frozen=True)
class CloudConfig:
    """Provider cloud LLM always-on (analyzer + composer) — Fase 22.

    Distinto da CloudBurst (on-demand hotkey): qui il modello cloud rimpiazza
    Ollama come provider LLM PRIMARIO. Si attiva impostando PROVIDERS.llm = "cloud".

    Mapping model: il caller passa un model string (es. "llama3.2:3b" per analyzer,
    "qwen2.5:7b-instruct" per composer). Il provider lo rimappa al model cloud
    corrispondente via `analyzer_remap` / `composer_remap` / `default_remap`.

    Vantaggi vs Ollama VM CPU:
    - 10-50x latenza inferiore (Haiku 4.5 ~500ms vs Qwen 7b CPU ~5-10s)
    - Qualità top di mercato
    - Mac libero di altre risorse (no Ollama locale, no Ollama VM in uso)

    Costo: pay-per-token. Vedi CloudBurstConfig per pricing simile.
    """
    # Provider: "anthropic" | "openai" | "deepseek"
    provider: str = "anthropic"
    api_key_env: str = "ANTHROPIC_API_KEY"

    # Mapping: model string passato dal caller → modello cloud effettivo
    # Configurabile per ottimizzare costo/qualità per ogni stage.
    analyzer_remap: str = "claude-haiku-4-5"          # fast + cheap per classifier
    composer_remap: str = "claude-sonnet-4-6"         # qualità top per generation
    default_remap: str = "claude-haiku-4-5"

    # Pattern matching per identificare quale "ruolo" è una chiamata:
    # se model contiene una di queste keyword, usa il remap analyzer
    analyzer_keywords: tuple[str, ...] = ("llama3.2:3b", "phi3", "qwen2.5:3b", "haiku")
    composer_keywords: tuple[str, ...] = ("qwen2.5:7b", "llama3.1:8b", "mistral:7b",
                                          "qwen2.5:14b", "sonnet", "opus")

    # Reasoning mode (Fase 19 pattern) per il composer
    composer_reasoning_enabled: bool = False    # off per default, costa di più
    composer_reasoning_budget: int = 3000

    request_timeout_s: float = 60.0


# ----- Multimodal image retrieval (Fase 20: CLIP/SigLIP) -----
@dataclass(frozen=True)
class ImageConfig:
    """Multimodal image layer: estrae slide PPTX come PNG, embedda con CLIP,
    indicizza per retrieval visivo. Opt-in.

    Use case: l'utente dice "lo slide con il diagramma di flusso del processo
    onboarding". Il vector search testuale fa fatica. CLIP/SigLIP allineano
    embedding testo↔immagine: la query trova la slide direttamente.

    Stack: extraction via LibreOffice headless → PDF → PNG, embedding via
    sentence-transformers/clip-ViT-B-32 (open weights, ~340MB).
    """
    enabled: bool = False

    # CLIP model — sentence-transformers ID. Multilingual:
    #   "clip-ViT-B-32"                       → English-focused, ~340MB
    #   "sentence-transformers/clip-ViT-B-32-multilingual-v1"  → multilingue (IT/EN), ~860MB
    #   "google/siglip-base-patch16-224"      → SigLIP migliore qualità
    embed_model: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

    # DPI rendering slide → PNG. 150 = qualità OK, file ~150KB/slide.
    render_dpi: int = 150

    # Top-K immagini retrievate per query.
    top_k: int = 4

    # Soglia minima cosine per ritornare un'immagine.
    min_similarity: float = 0.20


# ----- Cloud Burst (Fase 18+19: cloud LLM + reasoning on-demand) -----
@dataclass(frozen=True)
class CloudBurstConfig:
    """Cloud LLM premium per query high-stakes (opt-in).

    Filosofia: il flusso principale resta local-first. Quando l'utente preme
    una hotkey dedicata (Cmd+Shift+B) DURANTE una call, il sistema "burst"
    chiama un modello cloud premium (Claude Opus / GPT-4o / o3) con il
    contesto completo: trascrizione recente + chunks KB + memorie. Risposta
    streaming in overlay.

    Privacy warning: questo manda trascrizione + KB chunks al provider cloud.
    L'utente DEVE attivarlo esplicitamente e capire le implicazioni.
    """
    enabled: bool = False

    # Provider: "anthropic" | "openai" | "deepseek"
    provider: str = "anthropic"

    # API key — leggere da env var (mai hardcoded).
    # ANTHROPIC_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY
    api_key_env: str = "ANTHROPIC_API_KEY"

    # Modello specifico:
    #   anthropic: claude-opus-4-6, claude-sonnet-4-6 (reasoning), claude-haiku-4-5
    #   openai:    gpt-4o, gpt-4.1, o3, o3-mini
    #   deepseek:  deepseek-reasoner (R1)
    model: str = "claude-sonnet-4-6"

    # Modalità reasoning (Fase 19): richiede al modello catena di ragionamento
    # estesa. Compatibile con claude sonnet/opus + o3/o3-mini + deepseek-r1.
    # Costa di più (token thinking interni fatturati) ma qualità top per
    # query analitiche complesse ("riassumi pro/contro emersi nei 30 min").
    reasoning_enabled: bool = True

    # Budget thinking tokens (Anthropic: thinking budget; OpenAI o3: reasoning effort).
    # Aumentalo per query veramente complesse. ~5000 = ~4-8s pensiero.
    reasoning_budget_tokens: int = 5000

    # Max output tokens
    max_output_tokens: int = 2000

    # Timeout chiamata cloud — reasoning models possono richiedere 10-30s.
    request_timeout_s: float = 60.0

    # Hotkey: tradotto runtime in pyobjc keymask
    hotkey_label: str = "Cmd+Shift+B"


# ----- Server (Fase 12: cloud-ready FastAPI deployment) -----
@dataclass(frozen=True)
class ServerConfig:
    """FastAPI server config (Fase 12).

    Quando un componente in PROVIDERS è "remote", il client istanzia un
    RemoteXProvider che parla con questo server via REST/SSE.

    Side-server: lo stesso codice degli adapter local viene esposto come servizio
    HTTP. In questo modo un team può deployare 1 server con GPU + Ollama,
    e N client (laptop) lo usano via API key.

    NB: tutte le rotte sono autenticate via Bearer token. Senza token = 401.
    """
    # ── Server-side (chi lancia uvicorn) ──
    host: str = "0.0.0.0"
    port: int = 8787

    # API key richiesta nel header `Authorization: Bearer <key>`.
    # In dev usare default. In prod sostituire via env MEETING_COPILOT_API_KEY.
    api_key: str = "dev-secret-change-me"

    # CORS: domini autorizzati. ["*"] in dev, restringi in prod.
    cors_origins: tuple[str, ...] = ("*",)

    # Timeout per ogni richiesta lato server (es. LLM lento, embedding batch grande).
    request_timeout_s: float = 120.0

    # ── Client-side (chi setta PROVIDERS.*="remote") ──
    # Base URL del server. Override via env MEETING_COPILOT_SERVER_URL.
    client_base_url: str = "http://localhost:8787"

    # Timeout della singola HTTP request (deve essere ≥ request_timeout_s).
    client_timeout_s: float = 130.0

    # Retry automatico in caso di 5xx o connection error (max N volte).
    client_max_retries: int = 2


# ----- GraphRAG (Fase 17: knowledge graph layer) -----
@dataclass(frozen=True)
class GraphConfig:
    """Knowledge Graph layer (entity-relation) per neighborhood expansion.

    Architettura lite (no networkx / no Louvain): SQLite + embeddings.
    Quando attivo, post-indexing un LLM estrae entità (named entities + concetti)
    e relazioni dai chunks. Al retrieval, una query embeddiata trova entità
    rilevanti; il sistema recupera i chunks dei vicini 1-hop e li fonde via
    RRF con il retrieval vettoriale standard.

    Benefici (Microsoft GraphRAG paper, Tao Yu et al., aprile 2025):
    - +25-40% recall su query "globali" o multi-hop
    - Migliora il grounding su entità rare/nuove
    - Robusto a typo nei chunk source (embedding entità è normalizzante)

    Costo: ~1 LLM call per chunk durante indexing (one-shot, parallelizzabile).
    Inference: +50-150ms per query (1 SQL lookup + 1 embed match).
    """
    # Master switch — opt-in. Quando OFF il graph non viene né costruito né usato.
    enabled: bool = False

    # Modello LLM per extraction. llama3.2:3b veloce; qwen2.5:7b-instruct qualità.
    extractor_model: str = "llama3.2:3b"

    # Max entità + relazioni estratte per chunk. Cap per evitare prompt giganti.
    max_entities_per_chunk: int = 10
    max_relations_per_chunk: int = 8

    # Cosine similarity threshold per fondere entità duplicate (es. "ProductA"
    # e "ProductA" → stessa entità). Above = merge, below = entità separata.
    entity_merge_threshold: float = 0.88

    # Hop expansion al retrieval: 1 = vicini diretti, 2 = vicini-di-vicini.
    # 2 esplode la lista candidati ~quadraticamente — usare solo per KB piccole.
    expansion_hops: int = 1

    # Max chunks aggiuntivi recuperati dal grafo (over and above il dense+sparse).
    max_graph_chunks: int = 6

    # Peso RRF della terza fonte "graph" oltre dense+sparse.
    # Default: 0.2 (la maggior parte del segnale resta dense/sparse).
    rrf_graph_weight: float = 0.2

    # Skip embedding per entità che sono numeri puri o stringhe troppo corte.
    min_entity_len: int = 2


# ----- Memory (Fase 15: episodic cross-call memory) -----
@dataclass(frozen=True)
class MemoryConfig:
    # Master switch — quando OFF nessuna estrazione e nessun retrieval.
    enabled: bool = True

    # Max facts che l'extractor estrae per session (per non saturare la KB).
    max_facts_per_session: int = 15

    # Quante memorie max al composer per query.
    max_memories_per_query: int = 5

    # Min cosine similarity per considerare una memory rilevante.
    min_similarity: float = 0.35

    # Default scope quando non c'è KB attiva (fallback).
    default_scope: str = "general"


# Singleton istanze
AUDIO = AudioConfig()
VAD = VADConfig()
LLM = LLMConfig()
WATCHER = WatcherConfig()
TELEMETRY = TelemetryConfig()
MEMORY = MemoryConfig()
GRAPH = GraphConfig()
SERVER = ServerConfig()
CLOUD_BURST = CloudBurstConfig()
IMAGE = ImageConfig()
CLOUD = CloudConfig()
BUDGET = BudgetConfig()
SESSION = SessionConfig()


# ──────────────────────────────────────────────────────────────
# Provider / STT / RAG defaults — override per BUNDLED mode
# ──────────────────────────────────────────────────────────────
# In dev mode i defaults sono "local" (Ollama Mac + ChromaDB local etc.) e tu
# (Matteo) applichi i sed per cambiarli a cloud/remote.
# In bundled mode (.app friend distribution) la roba local NON è disponibile
# (torch/ollama/mlx esclusi per ridurre size 1.2GB → 305MB). Quindi forziamo
# i defaults a cloud/remote profile — friend deve avere:
#   - Anthropic API key (LLM cloud)
#   - Deepgram API key (STT cloud)
#   - Tailscale + URL VM OurCo + Bearer (per embed/vector/rerank remote)
if _IS_BUNDLED:
    print("[config] bundled mode: cloud-friend profile (LLM=cloud, STT=deepgram, vector=remote)")
    PROVIDERS = ProvidersConfig(
        stt="cloud",          # Deepgram (cloud)
        embedding="remote",   # Server VM via Tailscale
        llm="cloud",          # Anthropic
        vector_store="remote",  # Server VM ChromaDB
        rerank="remote",      # Server VM (disabilitato runtime via RAG.rerank_enabled=False)
        sparse_store="remote",  # Server VM BM25
    )
    STT = STTConfig(
        backend="deepgram",       # STT cloud Deepgram
        streaming_enabled=True,    # WebSocket live
        # Altri default invariati
    )
    # Override RAG per profilo cloud-friend
    RAG = RAGConfig(
        min_score_threshold=0.001,   # permissivo (era 0.15)
        rerank_enabled=False,        # rerank disabilitato (server CCX23 CPU troppo lento)
        crag_enabled=False,          # disabled (extra LLM call)
        rrf_dense_weight=0.85,
        rrf_sparse_weight=0.15,
        final_top_k=3,
    )
else:
    # Dev mode: defaults locali (tu applichi sed per overrides)
    STT = STTConfig()
    RAG = RAGConfig()
    PROVIDERS = ProvidersConfig()
