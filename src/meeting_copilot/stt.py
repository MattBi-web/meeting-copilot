"""Wrapper STT con due backend.

Backend supportati (selezionabili da config.STT.backend):
- "mlx"            → mlx-whisper. USA GPU + Neural Engine su Apple Silicon.
                     ~10x più veloce di faster-whisper su CPU. RACCOMANDATO su M-series.
- "faster-whisper" → CTranslate2 su CPU. Fallback per macchine non-Apple-Silicon
                     o quando MLX non è disponibile.

Espone un'unica funzione `transcribe(audio: np.ndarray) -> Transcription` che astrae
il backend scelto. Il modello è caricato lazily al primo `transcribe()`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import AUDIO, STT


@dataclass(frozen=True)
class Transcription:
    text: str
    language: str
    duration_s: float


# ─────────────────────────────────────────────────────────────
# Modelli MLX su Hugging Face (mlx-community)
# ─────────────────────────────────────────────────────────────
_MLX_MODEL_MAP = {
    "tiny":            "mlx-community/whisper-tiny-mlx",
    "base":            "mlx-community/whisper-base-mlx",
    "small":           "mlx-community/whisper-small-mlx",
    "medium":          "mlx-community/whisper-medium-mlx",
    "large-v3":        "mlx-community/whisper-large-v3-mlx",
    "turbo":           "mlx-community/whisper-large-v3-turbo",
    "large-v3-turbo":  "mlx-community/whisper-large-v3-turbo",
}


def _resolve_mlx_repo(model_size: str) -> str:
    """Mappa il nome breve a un repo Hugging Face mlx-community."""
    if model_size.startswith("mlx-community/") or "/" in model_size:
        return model_size  # già un repo HF completo
    return _MLX_MODEL_MAP.get(model_size, f"mlx-community/whisper-{model_size}-mlx")


# ─────────────────────────────────────────────────────────────
# Backend "mlx-whisper"
# ─────────────────────────────────────────────────────────────
_mlx_repo: str | None = None
_mlx_warmed = False


def _init_mlx() -> None:
    """Carica/cache il modello mlx-whisper. mlx-whisper gestisce la cache
    del modello interna al processo, quindi ci basta una transcribe() di
    warmup per scaricare e tenere il modello in memoria."""
    global _mlx_repo, _mlx_warmed
    if _mlx_warmed:
        return

    try:
        import mlx_whisper
    except ImportError as e:
        raise RuntimeError(
            "mlx-whisper non installato. Esegui: uv pip install -e .\n"
            f"({e})"
        ) from e

    _mlx_repo = _resolve_mlx_repo(STT.model_size)
    print(f"[stt] backend=mlx, model={_mlx_repo}")
    print("[stt] warming up MLX model (download al primo run)…")

    # Pre-warming: 1 secondo di silenzio
    warmup_audio = np.zeros(AUDIO.sample_rate, dtype=np.float32)
    _ = mlx_whisper.transcribe(
        warmup_audio,
        path_or_hf_repo=_mlx_repo,
        language=STT.language or "en",
        condition_on_previous_text=False,
        verbose=None,    # None = silenzia anche la TQDM progress bar
    )
    _mlx_warmed = True
    print("[stt] MLX model ready (loaded on GPU + Neural Engine).")


def _transcribe_mlx(segment: np.ndarray) -> Transcription:
    import mlx_whisper

    duration = segment.shape[0] / AUDIO.sample_rate

    result = mlx_whisper.transcribe(
        segment,
        path_or_hf_repo=_mlx_repo,
        language=STT.language,           # None = auto, "it"/"en" = forzato
        condition_on_previous_text=STT.condition_on_previous_text,
        no_speech_threshold=0.6,         # filtra segmenti rumorosi
        verbose=None,    # None = silenzia anche la TQDM progress bar
    )

    text = (result.get("text") or "").strip()
    language = result.get("language", "unknown") or "unknown"

    return Transcription(text=text, language=language, duration_s=duration)


# ─────────────────────────────────────────────────────────────
# Backend "faster-whisper" (CPU, fallback)
# ─────────────────────────────────────────────────────────────
_fw_model = None


def _init_faster_whisper():
    global _fw_model
    if _fw_model is not None:
        return _fw_model

    from faster_whisper import WhisperModel

    print(
        f"[stt] backend=faster-whisper, model={STT.model_size} "
        f"({STT.device}/{STT.compute_type}, threads={STT.cpu_threads})"
    )
    _fw_model = WhisperModel(
        STT.model_size,
        device=STT.device,
        compute_type=STT.compute_type,
        cpu_threads=STT.cpu_threads,
    )
    print("[stt] warming up faster-whisper model…")
    warmup_audio = np.zeros(AUDIO.sample_rate, dtype=np.float32)
    _ = list(
        _fw_model.transcribe(
            warmup_audio,
            language=STT.language or "en",
            beam_size=STT.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
        )[0]
    )
    print("[stt] faster-whisper model ready.")
    return _fw_model


def _transcribe_fw(segment: np.ndarray) -> Transcription:
    model = _init_faster_whisper()
    duration = segment.shape[0] / AUDIO.sample_rate

    segments_iter, info = model.transcribe(
        segment,
        language=STT.language,
        beam_size=STT.beam_size,
        vad_filter=STT.vad_filter,
        condition_on_previous_text=STT.condition_on_previous_text,
    )
    parts = [s.text.strip() for s in segments_iter]
    text = " ".join(p for p in parts if p).strip()

    return Transcription(
        text=text,
        language=info.language if info else "unknown",
        duration_s=duration,
    )


# ─────────────────────────────────────────────────────────────
# API pubblica
# ─────────────────────────────────────────────────────────────
def warmup() -> None:
    """Pre-carica il modello. Va chiamato all'avvio prima di entrare nel
    loop principale, così l'utente non paga il warmup sulla prima frase.

    Per backend "deepgram"/cloud → delega al provider factory (no model locale).
    """
    if STT.backend == "mlx":
        _init_mlx()
    elif STT.backend == "faster-whisper":
        _init_faster_whisper()
    elif STT.backend in ("deepgram", "voxtral"):
        # Cloud / external: delega al provider abstraction
        from .providers import get_stt_provider
        p = get_stt_provider()
        p.warmup()
    else:
        raise ValueError(f"backend STT sconosciuto: {STT.backend!r}")


def transcribe(segment: np.ndarray) -> Transcription:
    """Trascrive un segmento audio mono float32 @ 16 kHz.

    Args:
        segment: np.ndarray shape (N,), float32.

    Returns:
        Transcription. text può essere "" se Whisper non ha trovato parlato.
    """
    if STT.backend == "mlx":
        if not _mlx_warmed:
            _init_mlx()
        return _transcribe_mlx(segment)
    elif STT.backend == "faster-whisper":
        return _transcribe_fw(segment)
    elif STT.backend in ("deepgram", "voxtral"):
        # Cloud / external: delega al provider abstraction
        from .providers import get_stt_provider
        p = get_stt_provider()
        return p.transcribe(segment)
    else:
        raise ValueError(f"backend STT sconosciuto: {STT.backend!r}")
