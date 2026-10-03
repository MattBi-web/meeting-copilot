"""Speaker diarization euristica (no ML model).

Sfrutta il fatto che il nostro Aggregate Device 'MeetingCapture' espone canali separati:
- canali 0-1 → BlackHole 2ch (audio di sistema = controparte)
- canale 2   → microfono fisico = TU

Confrontando l'energia RMS sui due gruppi capiamo chi sta parlando:
- self    → solo mic ha energia
- other   → solo BlackHole ha energia
- both    → entrambi (sovrapposizione, overlap parlato)
- silence → nessuno dei due

Questo è un suppressore critico per il Watcher Agent: NON suggerire mentre TU parli.

Funziona su segmenti grezzi multi-canale (PRIMA del downmix mono).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


SpeakerLabel = Literal["self", "other", "both", "silence", "unknown"]


@dataclass(frozen=True)
class SpeakerEstimate:
    label: SpeakerLabel
    self_rms: float          # 0..1
    other_rms: float         # 0..1
    confidence: float        # 0..1


# Thresholds (calibrate sui tuoi livelli — possono essere tunati)
_SILENCE_THRESHOLD = 0.005      # RMS sotto questo = silenzio per quel canale
_DOMINANCE_RATIO = 2.0          # un canale è "dominante" se RMS >= 2x dell'altro


def classify_speaker(multichannel_chunk: np.ndarray, n_blackhole_ch: int = 2) -> SpeakerEstimate:
    """Classifica un chunk audio MULTI-CANALE (pre-downmix) in self/other/both/silence.

    Args:
        multichannel_chunk: np.ndarray shape (N, C), C = numero canali totali.
                            Convenzione attuale MeetingCapture: 3 canali
                            [BlackHole_L, BlackHole_R, Mic].
        n_blackhole_ch: quanti canali iniziali appartengono a BlackHole (default 2 stereo).

    Returns:
        SpeakerEstimate con label + RMS per gruppo.
    """
    if multichannel_chunk.ndim == 1 or multichannel_chunk.size == 0:
        # Già mono o vuoto: non possiamo diarizzare
        return SpeakerEstimate("unknown", 0.0, 0.0, 0.0)

    n_channels = multichannel_chunk.shape[1]
    if n_channels < 2:
        return SpeakerEstimate("unknown", 0.0, 0.0, 0.0)

    n_bh = min(n_blackhole_ch, n_channels - 1)
    if n_bh < 1:
        return SpeakerEstimate("unknown", 0.0, 0.0, 0.0)

    # Energia per gruppo
    bh_signal = multichannel_chunk[:, :n_bh].mean(axis=1)
    mic_signal = multichannel_chunk[:, n_bh:].mean(axis=1)

    other_rms = float(np.sqrt(np.mean(bh_signal ** 2)))
    self_rms = float(np.sqrt(np.mean(mic_signal ** 2)))

    self_active = self_rms >= _SILENCE_THRESHOLD
    other_active = other_rms >= _SILENCE_THRESHOLD

    if not self_active and not other_active:
        return SpeakerEstimate("silence", self_rms, other_rms, 0.8)

    if self_active and not other_active:
        return SpeakerEstimate("self", self_rms, other_rms,
                               confidence=_score(self_rms, other_rms))

    if other_active and not self_active:
        return SpeakerEstimate("other", self_rms, other_rms,
                               confidence=_score(other_rms, self_rms))

    # Entrambi attivi → ratio decide
    if self_rms >= other_rms * _DOMINANCE_RATIO:
        return SpeakerEstimate("self", self_rms, other_rms,
                               confidence=_score(self_rms, other_rms) * 0.8)
    if other_rms >= self_rms * _DOMINANCE_RATIO:
        return SpeakerEstimate("other", self_rms, other_rms,
                               confidence=_score(other_rms, self_rms) * 0.8)

    # Davvero entrambi (overlap)
    return SpeakerEstimate("both", self_rms, other_rms, confidence=0.5)


def _score(dom: float, sub: float) -> float:
    """Confidence in base a quanto è dominante il canale principale."""
    if dom <= 0:
        return 0.0
    ratio = dom / max(sub, 1e-6)
    # ratio 1 → 0.5; ratio 5 → ~0.9
    return float(min(1.0, 0.5 + 0.1 * (ratio - 1.0)))


# ──────────────────────────────────────────────────────────────
# Aggregator: classifica un SEGMENTO completo (più chunk) → label finale
# ──────────────────────────────────────────────────────────────
def aggregate_speaker_label(estimates: list[SpeakerEstimate]) -> SpeakerLabel:
    """Dato una lista di stime (una per chunk del segmento), decide
    il label dominante per l'intero segmento.

    Logica: maggioranza ponderata sulla confidence."""
    if not estimates:
        return "unknown"
    weights: dict[str, float] = {}
    for est in estimates:
        weights[est.label] = weights.get(est.label, 0.0) + est.confidence
    # Ignora "silence" se ci sono altre label significative
    sorted_labels = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    if sorted_labels[0][0] == "silence" and len(sorted_labels) > 1 and sorted_labels[1][1] > 0.3:
        return sorted_labels[1][0]  # type: ignore
    return sorted_labels[0][0]  # type: ignore
