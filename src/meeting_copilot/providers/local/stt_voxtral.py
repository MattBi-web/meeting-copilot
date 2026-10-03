"""Voxtral STT provider — STUB (Fase 9.1.B).

Stato a maggio 2026:
─────────────────────
Mistral ha rilasciato:
- Voxtral Mini 3B (luglio 2025) — chunked transcription, batch
- Voxtral Realtime (febbraio 2026) — encoder causale streaming, cloud API

Per uso ON-DEVICE su Apple Silicon esiste `mlx-voxtral` (mzbac), ma con
3 problemi seri:
  1. License: "Personal Use Only" → blocca commerce
  2. NON è streaming token-by-token (è chunked come Whisper)
  3. Qualità Voxtral Mini 3B = circa Whisper Large-v3 (no salto qualitativo)

Conclusione: Whisper Turbo MLX resta la scelta migliore per noi.
Voxtral local diventerà interessante quando:
- Mistral rilascerà MLX port ufficiale con license commerciale
- Oppure noi accetteremo cloud (Mistral La Plateforme) violando local-first

Questo stub esiste per:
- Documentare la decisione architettonica (per il futuro)
- Permettere swap rapido via 1 flag config quando le condizioni cambiano

Quando vorrai attivarlo davvero:
1. Verifica license Mistral ufficiale per uso commerciale
2. `pip install mlx-voxtral` (o port ufficiale Mistral, se disponibile)
3. Implementa i metodi sotto sostituendo `NotImplementedError`
4. In config.py: `STT.backend = "voxtral"`
"""
from __future__ import annotations

import numpy as np

from ...config import AUDIO, STT
from ..base import Transcription


# Modelli Voxtral disponibili (Mistral + community MLX ports)
_VOXTRAL_MODELS = {
    "voxtral-mini-3b":             "mistralai/Voxtral-Mini-3B-2507",
    "voxtral-mini-3b-4bit":        "mzbac/voxtral-mini-3b-4bit-mixed",   # quantizzato MLX
    "voxtral-mini-3b-8bit":        "mzbac/voxtral-mini-3b-8bit",
    "voxtral-realtime-cloud":      "mistral-la-plateforme",              # placeholder
}


class LocalVoxtralSTTProvider:
    """STUB. Implementazione futura quando saranno presenti i prerequisiti
    di licenza + qualità streaming.

    Per ora, qualsiasi chiamata solleva NotImplementedError con istruzioni
    chiare su cosa fare per attivarlo davvero."""

    def __init__(self):
        self._model_id = STT.model_size

    # ── STTProvider Protocol ──

    def warmup(self) -> None:
        raise NotImplementedError(
            "Voxtral STT provider è uno stub.\n"
            "Stato attuale: Whisper Turbo MLX è la scelta migliore per Apple Silicon local.\n"
            "Per attivare Voxtral:\n"
            "  1. Verifica license Mistral commerciale (mlx-voxtral community è 'Personal Use Only')\n"
            "  2. `pip install mlx-voxtral` + `pip install git+https://github.com/huggingface/transformers`\n"
            "  3. Implementa i metodi in providers/local/stt_voxtral.py\n"
            "  4. In config.py: STT.backend = 'voxtral'\n"
            "Per ora, lascia STT.backend = 'mlx' (default)."
        )

    def transcribe(self, segment: np.ndarray) -> Transcription:
        # Stesso messaggio del warmup
        self.warmup()
        # Non raggiunto:
        return Transcription(text="", language="unknown", duration_s=0.0)


# ──────────────────────────────────────────────────────────────
# Placeholder implementation skeleton (per quando faremo l'integrazione vera).
# Commentato perché solleverebbe ImportError senza mlx-voxtral.
# ──────────────────────────────────────────────────────────────
# def _build_real_implementation_sketch():
#     """
#     Schema della futura implementazione (da abilitare quando mlx-voxtral
#     ha license commerciale o quando Mistral rilascia port ufficiale):
#
#     from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor
#     model = VoxtralForConditionalGeneration.from_pretrained(_VOXTRAL_MODELS[id])
#     processor = VoxtralProcessor.from_pretrained(_VOXTRAL_MODELS[id])
#
#     # Warmup dummy
#     dummy = np.zeros(AUDIO.sample_rate, dtype=np.float32)
#     inputs = processor.apply_transcrition_request(
#         language=STT.language or "en", audio=dummy,
#     )
#     _ = model.generate(**inputs, max_new_tokens=1, temperature=0.0)
#
#     # Transcribe
#     inputs = processor.apply_transcrition_request(
#         language=STT.language or "en", audio=segment,
#     )
#     outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.0)
#     text = processor.decode(
#         outputs[0][inputs.input_ids.shape[1]:],
#         skip_special_tokens=True,
#     )
#     return Transcription(text=text.strip(), language=info.language, duration_s=duration)
#     """
#     pass
