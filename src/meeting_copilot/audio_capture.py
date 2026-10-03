"""Audio capture dall'Aggregate Device 'MeetingCapture'.

Espone due generators:
- `audio_stream()`        → chunks mono float32 16 kHz (backward compat)
- `audio_stream_with_speaker()` → chunks mono + label speaker (self/other/both/silence)
                                  ricavato dal raw multichannel via speaker.classify_speaker()
"""
from __future__ import annotations

import argparse
import queue
import sys
from collections.abc import Iterator

import numpy as np
import sounddevice as sd

from .config import AUDIO


def list_devices() -> None:
    """Stampa tutti i device audio disponibili. Usalo per verificare che
    'MeetingCapture' sia presente."""
    print("Available audio devices:\n")
    print(sd.query_devices())
    print(f"\nLooking for device named: {AUDIO.device_name!r}")
    try:
        idx = _find_device_index(AUDIO.device_name)
        info = sd.query_devices(idx)
        print(f"\n✓ Found '{AUDIO.device_name}' (index {idx}):")
        print(f"  - max input channels: {info['max_input_channels']}")
        print(f"  - default sample rate: {info['default_samplerate']}")
    except RuntimeError as e:
        print(f"\n✗ {e}")
        print("Hai completato la Fase 0 (SETUP.md)?")


def _find_device_index(name: str) -> int:
    """Cerca un device per substring nel nome. Case-insensitive."""
    target = name.lower()
    for i, dev in enumerate(sd.query_devices()):
        if target in dev["name"].lower() and dev["max_input_channels"] > 0:
            return i
    raise RuntimeError(
        f"Device '{name}' non trovato fra gli input audio. "
        f"Verifica l'Aggregate Device in Audio MIDI Setup."
    )


def audio_stream_with_speaker(
    device_name: str | None = None,
    sample_rate: int | None = None,
    block_size: int | None = None,
) -> "Iterator[tuple[np.ndarray, object]]":
    """Versione estesa che yielda (mono_chunk, SpeakerEstimate) per chunk.

    Usa lo stesso flow ma la classificazione speaker avviene SUL CHUNK
    multichannel prima del downmix mono. Costo extra: ~0.1ms/chunk.
    """
    # Import locale per non forzare il modulo speaker su chi usa audio_stream()
    from .speaker import classify_speaker

    device_name = device_name or AUDIO.device_name
    sample_rate = sample_rate or AUDIO.sample_rate
    block_size = block_size or AUDIO.block_size

    device_idx = _find_device_index(device_name)
    info = sd.query_devices(device_idx)
    channels = min(int(info["max_input_channels"]), AUDIO.input_channels)

    audio_q: queue.Queue[tuple[np.ndarray, object]] = queue.Queue(maxsize=64)

    def callback(indata: np.ndarray, frames: int, time_, status) -> None:
        if status:
            print(f"[audio] status: {status}", file=sys.stderr)
        # Speaker classify SUL CHUNK MULTICHANNEL prima del downmix
        if indata.ndim == 2 and indata.shape[1] > 1:
            speaker_est = classify_speaker(indata)
            mono = indata.mean(axis=1)
        else:
            from .speaker import SpeakerEstimate
            speaker_est = SpeakerEstimate("unknown", 0.0, 0.0, 0.0)
            mono = indata[:, 0] if indata.ndim == 2 else indata
        try:
            audio_q.put_nowait((mono.copy(), speaker_est))
        except queue.Full:
            pass

    stream = sd.InputStream(
        device=device_idx,
        channels=channels,
        samplerate=sample_rate,
        blocksize=block_size,
        dtype="float32",
        callback=callback,
    )
    with stream:
        print(
            f"[audio] streaming (with speaker diarization) from device #{device_idx} "
            f"({info['name']!r}) @ {sample_rate} Hz, {channels} ch, block={block_size}",
            file=sys.stderr,
        )
        try:
            while True:
                yield audio_q.get()
        except KeyboardInterrupt:
            print("\n[audio] stop requested", file=sys.stderr)


def audio_stream(
    device_name: str | None = None,
    sample_rate: int | None = None,
    block_size: int | None = None,
) -> Iterator[np.ndarray]:
    """Generator: yields chunks mono float32 a `sample_rate`, `block_size` samples each.

    L'Aggregate Device 'MeetingCapture' espone 2 canali (mic + BlackHole).
    Li miscelermo a mono con la media: questo dà un singolo stream che contiene
    sia te sia gli altri, perfetto per la trascrizione (senza diarization per ora).
    """
    device_name = device_name or AUDIO.device_name
    sample_rate = sample_rate or AUDIO.sample_rate
    block_size = block_size or AUDIO.block_size

    device_idx = _find_device_index(device_name)
    info = sd.query_devices(device_idx)
    channels = min(int(info["max_input_channels"]), AUDIO.input_channels)

    # Coda thread-safe per passare audio dal callback al main thread
    audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)

    def callback(indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            # Overflow/underflow: log su stderr, NON bloccare lo stream
            print(f"[audio] status: {status}", file=sys.stderr)
        # indata shape: (frames, channels), dtype float32
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = indata.mean(axis=1)
        else:
            mono = indata[:, 0] if indata.ndim == 2 else indata
        try:
            audio_q.put_nowait(mono.copy())
        except queue.Full:
            # Sotto pressione: skip frame (preferibile a bloccare l'audio thread)
            pass

    stream = sd.InputStream(
        device=device_idx,
        channels=channels,
        samplerate=sample_rate,
        blocksize=block_size,
        dtype="float32",
        callback=callback,
    )

    with stream:
        print(
            f"[audio] streaming from device #{device_idx} ({info['name']!r}) "
            f"@ {sample_rate} Hz, {channels} ch, block={block_size} samples",
            file=sys.stderr,
        )
        try:
            while True:
                yield audio_q.get()
        except KeyboardInterrupt:
            print("\n[audio] stop requested", file=sys.stderr)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Audio capture utility.")
    parser.add_argument("--list", action="store_true", help="List audio devices.")
    parser.add_argument(
        "--probe",
        type=float,
        default=0.0,
        help="Acquisisci audio per N secondi e stampa RMS (test rapido).",
    )
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    if args.probe > 0:
        import time

        print(f"Probing audio per {args.probe}s da {AUDIO.device_name!r}…")
        rms_values: list[float] = []
        t0 = time.monotonic()
        for chunk in audio_stream():
            rms = float(np.sqrt(np.mean(chunk**2)))
            rms_values.append(rms)
            if time.monotonic() - t0 >= args.probe:
                break
        avg = float(np.mean(rms_values))
        peak = float(np.max(rms_values))
        print(f"RMS medio: {avg:.4f} — peak: {peak:.4f}")
        if avg < 1e-4:
            print(
                "⚠️  Segnale quasi nullo. Verifica che l'output sia su 'MeetingMonitor' "
                "e che il microfono non sia mutato."
            )
        else:
            print("✓ Audio captured.")
        return

    parser.print_help()


if __name__ == "__main__":
    _cli()
