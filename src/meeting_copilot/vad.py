"""Voice Activity Detection con silero-vad in streaming mode.

silero-vad lavora a 16 kHz su chunks da 512 samples (32 ms). Per ogni chunk
ritorna la probabilità che contenga parlato. Noi accumuliamo i chunks "vocali"
in un buffer, e quando troviamo silenzio prolungato (`min_silence_ms`) o il
buffer supera `max_segment_ms`, emettiamo un segmento da passare a Whisper.

Espone due generators:
- `segmenter(audio_chunks)`           → yields np.ndarray (mono segment)
- `segmenter_with_speaker(input)`     → yields (np.ndarray, dominant_speaker_label)
  dove `input` è un iterable di tuple (chunk_mono, SpeakerEstimate) come da
  `audio_capture.audio_stream_with_speaker()`.

Ogni segmento è float32 mono a 16 kHz, da `min_segment_ms` a `max_segment_ms`.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import torch
from silero_vad import load_silero_vad

from .config import AUDIO, VAD


_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_silero_vad()
    return _model


def segmenter(audio_chunks: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
    """Consuma un generator di chunks audio e produce segmenti vocali completi.

    Args:
        audio_chunks: iterable di np.ndarray shape (block_size,), float32, 16 kHz.

    Yields:
        np.ndarray shape (N,), float32, dove N >= min_segment_ms * 16
        ed N <= max_segment_ms * 16.
    """
    model = _get_model()
    sr = AUDIO.sample_rate
    block = AUDIO.block_size

    silence_blocks_needed = int(VAD.min_silence_ms / 1000 * sr / block)
    max_blocks = int(VAD.max_segment_ms / 1000 * sr / block)
    min_samples = int(VAD.min_segment_ms / 1000 * sr)

    buffer: list[np.ndarray] = []
    silence_count = 0
    in_speech = False

    for chunk in audio_chunks:
        if chunk.shape[0] != block:
            # Allinea: se chunk non è esattamente block samples,
            # silero-vad non gli darà la dimensione attesa. Per ora skippiamo.
            continue

        # silero-vad vuole un torch.Tensor 1D float32
        prob = float(model(torch.from_numpy(chunk), sr).item())

        is_speech = prob >= VAD.speech_threshold

        if is_speech:
            buffer.append(chunk)
            silence_count = 0
            in_speech = True
        else:
            if in_speech:
                # Stiamo registrando un segmento, conta i blocchi di silenzio
                buffer.append(chunk)  # includi un po' di silenzio per pulizia
                silence_count += 1

                if silence_count >= silence_blocks_needed:
                    # Fine segmento
                    seg = np.concatenate(buffer)
                    if seg.shape[0] >= min_samples:
                        yield _trim_trailing_silence(seg, silence_count, block)
                    buffer.clear()
                    silence_count = 0
                    in_speech = False

        # Hard cap su segment length: evita segmenti troppo lunghi
        if in_speech and len(buffer) >= max_blocks:
            seg = np.concatenate(buffer)
            if seg.shape[0] >= min_samples:
                yield seg
            buffer.clear()
            silence_count = 0
            in_speech = False


def segmenter_with_speaker(
    audio_chunks_with_speaker: Iterable[tuple[np.ndarray, object]],
) -> Iterator[tuple[np.ndarray, str]]:
    """Wrapper attorno a segmenter() che propaga la label speaker dominante.

    Aggrega le label dei chunk vocali interni al segmento e ritorna la label
    finale (self/other/both/silence/unknown).
    """
    from .speaker import aggregate_speaker_label

    model = _get_model()
    sr = AUDIO.sample_rate
    block = AUDIO.block_size

    silence_blocks_needed = int(VAD.min_silence_ms / 1000 * sr / block)
    max_blocks = int(VAD.max_segment_ms / 1000 * sr / block)
    min_samples = int(VAD.min_segment_ms / 1000 * sr)

    buffer: list[np.ndarray] = []
    speaker_estimates: list[object] = []
    silence_count = 0
    in_speech = False

    for chunk, speaker_est in audio_chunks_with_speaker:
        if chunk.shape[0] != block:
            continue

        prob = float(model(torch.from_numpy(chunk), sr).item())
        is_speech = prob >= VAD.speech_threshold

        if is_speech:
            buffer.append(chunk)
            speaker_estimates.append(speaker_est)
            silence_count = 0
            in_speech = True
        else:
            if in_speech:
                buffer.append(chunk)
                speaker_estimates.append(speaker_est)
                silence_count += 1
                if silence_count >= silence_blocks_needed:
                    seg = np.concatenate(buffer)
                    if seg.shape[0] >= min_samples:
                        seg_trimmed = _trim_trailing_silence(seg, silence_count, block)
                        label = aggregate_speaker_label(speaker_estimates)
                        yield (seg_trimmed, label)
                    buffer.clear()
                    speaker_estimates.clear()
                    silence_count = 0
                    in_speech = False

        if in_speech and len(buffer) >= max_blocks:
            seg = np.concatenate(buffer)
            if seg.shape[0] >= min_samples:
                label = aggregate_speaker_label(speaker_estimates)
                yield (seg, label)
            buffer.clear()
            speaker_estimates.clear()
            silence_count = 0
            in_speech = False


def _trim_trailing_silence(
    segment: np.ndarray, silence_blocks: int, block_size: int
) -> np.ndarray:
    """Rimuove il silenzio finale (lasciando 200 ms di coda per Whisper)."""
    keep_silence_samples = int(0.2 * AUDIO.sample_rate)
    silence_samples = silence_blocks * block_size
    cut = max(0, silence_samples - keep_silence_samples)
    if cut <= 0:
        return segment
    end = max(0, segment.shape[0] - cut)
    return segment[:end]
