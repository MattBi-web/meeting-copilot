"""Deepgram STT provider — opzione cloud per trascrizione.

Pattern: REST batch (pre-recorded API). Ogni segmento audio post-VAD
viene mandato a Deepgram come PCM 16-bit raw. Più semplice del WebSocket
streaming + sufficiente per il nostro use case (segmenti già pre-tagliati
da SileroVAD).

Quando attivare:
- Vuoi qualità top di mercato (~5% WER su parlato pulito IT/EN)
- Vuoi liberare CPU/RAM Mac (no MLX Whisper locale)
- Sei OK col pay-per-use (~$0.0043/min = €0.26/h)

Env var:
- DEEPGRAM_API_KEY (richiesta)

Models supportati:
- nova-3 (default): top quality, IT/EN nativi
- nova-2: bilanciato, leggermente più rapido
- enhanced: legacy ma cheap

Latency tipica: 150-400ms per segmento di 1-4 secondi.

NB: non implementa warmup (nessun modello locale).
"""
from __future__ import annotations

import io
import os
import wave
from dataclasses import dataclass

import numpy as np

from ...config import STT
from ..base import Transcription


@dataclass(frozen=True)
class DeepgramConfig:
    """Sotto-config Deepgram. Letta da config.STT.deepgram_* attributes."""
    model: str = "nova-3"
    language: str | None = "it"
    smart_format: bool = True
    punctuate: bool = True
    profanity_filter: bool = False
    diarize: bool = False  # se True, ritorna chi parla (utenti diversi)
    keywords: list[str] | None = None  # boost keywords (analogo initial_prompt)


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Converti numpy array float32 [-1, 1] in WAV mono 16-bit PCM."""
    # Float32 → int16
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _resolve_api_key() -> str:
    key = os.environ.get("DEEPGRAM_API_KEY", "")
    if not key:
        raise RuntimeError(
            "DEEPGRAM_API_KEY non impostata. Setta env var con la tua key:\n"
            "  export DEEPGRAM_API_KEY='sk_...'\n"
            "Get una key gratis su https://console.deepgram.com (200$ free credits)."
        )
    return key


def _load_vocabulary() -> list[str]:
    """Legge session_vocabulary.txt e ritorna parole come 'keywords' boost.

    Deepgram supporta keywords boosting (massimo ~50 keyword per request,
    con peso 1-10). Le passiamo come param `&keywords=word1:5&keywords=word2:3...`
    """
    try:
        from ...config import PROJECT_ROOT
        vocab_path = PROJECT_ROOT / "session_vocabulary.txt"
        if not vocab_path.exists():
            return []
        text = vocab_path.read_text(encoding="utf-8")
        # Splitta su spazi/newlines, dedupe, max 50 parole, no stopwords
        words: list[str] = []
        seen: set[str] = set()
        for w in text.split():
            w = w.strip()
            if len(w) < 3 or len(w) > 30:
                continue
            wl = w.lower()
            if wl in seen:
                continue
            seen.add(wl)
            words.append(w)
            if len(words) >= 50:
                break
        return words
    except Exception:
        return []


class DeepgramSTTProvider:
    """STTProvider via Deepgram REST batch API.

    Implementa il Protocol STTProvider (warmup, transcribe).
    """

    def __init__(self, model: str | None = None):
        self._model = model or "nova-3"
        # Cached keywords (rilette solo al warmup, non a ogni transcribe)
        self._keywords: list[str] = []

    def warmup(self) -> None:
        """No-op: nessun modello locale. Pre-carica vocabulary se presente."""
        self._keywords = _load_vocabulary()
        if self._keywords:
            print(f"[deepgram] caricate {len(self._keywords)} keywords da vocabulary")
        # Verifica connessione veloce
        try:
            import httpx
            r = httpx.get(
                "https://api.deepgram.com/v1/projects",
                headers={"Authorization": f"Token {_resolve_api_key()}"},
                timeout=5,
            )
            if r.status_code == 200:
                print(f"[deepgram] API key valid, model={self._model}")
            else:
                print(f"[deepgram] WARNING: connection test status {r.status_code}")
        except Exception as e:
            print(f"[deepgram] warmup connection test failed: {e}")

    def transcribe(self, segment: np.ndarray) -> Transcription:
        """Trascrivi un segmento audio (numpy float32 mono 16kHz).

        Ritorna Transcription. Latency ~150-400ms per segmento.
        """
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError("httpx non installato") from e

        duration = float(len(segment)) / 16000.0

        # Audio → WAV bytes
        wav_bytes = _audio_to_wav_bytes(segment, sample_rate=16000)

        # Build query params
        params: dict[str, str] = {
            "model": self._model,
            "smart_format": "true",
            "punctuate": "true",
        }
        # Multi-language support
        mode = getattr(STT, "transcription_mode", "single")
        if mode == "multi":
            params["language"] = "multi"
        elif mode == "auto":
            params["detect_language"] = "true"
        elif STT.language:
            params["language"] = STT.language

        # Keyword boosting (se vocabulary presente).
        # Nova-3: usa `keyterm` (Nova-2/legacy: usa `keywords:<intensity>`).
        # Auto-switch in base al model.
        url = "https://api.deepgram.com/v1/listen"
        if self._keywords:
            from urllib.parse import urlencode
            if self._model.startswith("nova-3") or self._model.startswith("nova-4"):
                # Nuova API: keyterm=word (no intensity).
                # Max 50 keyterms per request.
                kw_pairs = [("keyterm", w) for w in self._keywords[:100]]
            else:
                # Legacy: keywords=word:intensity (1-10).
                kw_pairs = [("keywords", f"{w}:3") for w in self._keywords[:100]]
            base_params = list(params.items())
            query = urlencode(base_params + kw_pairs)
            url = f"{url}?{query}"
            params = None  # già nell'URL

        headers = {
            "Authorization": f"Token {_resolve_api_key()}",
            "Content-Type": "audio/wav",
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, headers=headers, content=wav_bytes,
                                   params=params)
            if resp.status_code >= 400:
                print(f"[deepgram] HTTP {resp.status_code}: {resp.text[:200]}")
                return Transcription(text="", language="unknown", duration_s=duration)
            data = resp.json()
        except Exception as e:
            print(f"[deepgram] transcribe error: {e}")
            return Transcription(text="", language="unknown", duration_s=duration)

        # Extract transcript dal payload Deepgram
        results = data.get("results", {})
        channels = results.get("channels", [])
        if not channels:
            return Transcription(text="", language="unknown", duration_s=duration)
        alternatives = channels[0].get("alternatives", [])
        if not alternatives:
            return Transcription(text="", language="unknown", duration_s=duration)
        text = (alternatives[0].get("transcript") or "").strip()
        detected_lang = channels[0].get("detected_language") or STT.language or "it"

        return Transcription(text=text, language=detected_lang, duration_s=duration)
