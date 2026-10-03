"""Local STT provider (mlx-whisper + faster-whisper fallback)."""
from __future__ import annotations

from collections import deque

import numpy as np

from ...config import AUDIO, VAD, STT
from ..base import Transcription


_MLX_MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def _resolve_mlx_repo(model_size: str) -> str:
    if "/" in model_size:
        return model_size
    return _MLX_MODEL_MAP.get(model_size, f"mlx-community/whisper-{model_size}-mlx")


class LocalSTTProvider:
    """Implementa STTProvider via mlx-whisper o faster-whisper."""

    def __init__(self):
        self._backend = STT.backend
        self._warmed = False
        self._mlx_repo: str | None = None
        self._fw_model = None

        # Fase 9.1.A — initial_prompt rolling buffer:
        # tiene le ultime N parole trascritte per dare contesto a Whisper
        # sul prossimo chunk. Riduce gli "errori di confine" quando una
        # frase è spezzata tra due chunks (ora più frequenti perché
        # max_segment_ms è sceso a 4000).
        self._recent_words: deque[str] = deque(maxlen=VAD.initial_prompt_words)

    def warmup(self) -> None:
        if self._warmed:
            return
        if self._backend == "mlx":
            self._warmup_mlx()
        else:
            self._warmup_fw()
        self._warmed = True

    def _warmup_mlx(self) -> None:
        import mlx_whisper
        self._mlx_repo = _resolve_mlx_repo(STT.model_size)
        print(f"[stt-local-mlx] warming up {self._mlx_repo}…")
        dummy = np.zeros(AUDIO.sample_rate, dtype=np.float32)
        _ = mlx_whisper.transcribe(
            dummy,
            path_or_hf_repo=self._mlx_repo,
            language=STT.language or "en",
            condition_on_previous_text=False,
            verbose=None,
        )
        print("[stt-local-mlx] ready (GPU + ANE).")

    def _warmup_fw(self) -> None:
        from faster_whisper import WhisperModel
        print(f"[stt-local-fw] warming up {STT.model_size} ({STT.device}/{STT.compute_type})…")
        self._fw_model = WhisperModel(
            STT.model_size,
            device=STT.device,
            compute_type=STT.compute_type,
            cpu_threads=STT.cpu_threads,
        )
        dummy = np.zeros(AUDIO.sample_rate, dtype=np.float32)
        _ = list(self._fw_model.transcribe(
            dummy,
            language=STT.language or "en",
            beam_size=STT.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
        )[0])
        print("[stt-local-fw] ready.")

    def transcribe(self, segment: np.ndarray) -> Transcription:
        if not self._warmed:
            self.warmup()
        duration = segment.shape[0] / AUDIO.sample_rate
        if self._backend == "mlx":
            tr = self._transcribe_mlx(segment, duration)
        else:
            tr = self._transcribe_fw(segment, duration)
        # Aggiorna rolling buffer (ultime N parole) per il prossimo chunk
        if tr.text:
            for w in tr.text.split():
                self._recent_words.append(w)
        return tr

    def _build_initial_prompt(self) -> str | None:
        """Costruisce il prompt iniziale combinando:
        1) Vocabolario custom della call (da session_vocabulary.txt) — termini
           rari, nomi, sigle che Whisper potrebbe mistranscribere
        2) Rolling buffer delle ultime N parole del chunk precedente (coerenza
           inter-chunk)
        Whisper lo usa come bias contestuale, NON come testo da emettere.
        """
        parts: list[str] = []
        # Custom vocab persistente (priorità alta per termini rari)
        try:
            from ...config import PROJECT_ROOT
            vocab_path = PROJECT_ROOT / "session_vocabulary.txt"
            if vocab_path.exists():
                vocab = vocab_path.read_text(encoding="utf-8").strip()
                if vocab:
                    parts.append(vocab)
        except Exception:
            pass
        # Rolling buffer
        if self._recent_words:
            parts.append(" ".join(self._recent_words))
        if not parts:
            return None
        # Whisper accetta max ~224 token come prompt
        joined = " ".join(parts)
        return joined[:1000]

    def _transcribe_mlx(self, segment: np.ndarray, duration: float) -> Transcription:
        import mlx_whisper
        kwargs = dict(
            path_or_hf_repo=self._mlx_repo,
            language=STT.language,
            condition_on_previous_text=STT.condition_on_previous_text,
            no_speech_threshold=0.6,
            verbose=None,
        )
        # Fase 9.1.A — initial_prompt per coerenza inter-chunk
        prompt = self._build_initial_prompt()
        if prompt:
            kwargs["initial_prompt"] = prompt
        result = mlx_whisper.transcribe(segment, **kwargs)
        return Transcription(
            text=(result.get("text") or "").strip(),
            language=result.get("language", "unknown") or "unknown",
            duration_s=duration,
        )

    def _transcribe_fw(self, segment: np.ndarray, duration: float) -> Transcription:
        segments_iter, info = self._fw_model.transcribe(
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
