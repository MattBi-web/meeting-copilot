"""Audio capture per Windows usando WASAPI loopback.

Mac: BlackHole virtual device + Aggregate Device
Windows: WASAPI loopback (built-in OS, zero install)

Pattern: sounddevice supporta WASAPI loopback con extra_settings WasapiSettings(loopback=True).
Cattura quello che sta uscendo da una qualsiasi output device (speakers / headphones).
Plus aggiungi il microfono via mixed stream o secondo capture parallelo.

Per il setup multi-source (system output + mic insieme) su Windows ci sono 3 opzioni:
1. **WASAPI loopback solo output**: cattura altri partecipanti, NON tua voce.
   Adatto se sei silente o se mic è captato da altro device fisico.
2. **WASAPI loopback output + mic ambient mix in software**: cattura entrambi,
   downmix software in mono.
3. **VB-Audio Cable virtual device**: setup come BlackHole su Mac, equivalente
   Aggregate Device.

Default qui: opzione 2 (loopback + mic mix). Funziona out-of-the-box senza driver custom.

NB: questo modulo è importato SOLO su sys.platform=='win32'. Su Mac usa audio_capture.py.
"""
from __future__ import annotations

import queue
import sys
from collections.abc import Iterator

import numpy as np

if sys.platform != "win32":
    raise ImportError("audio_capture_windows è solo per Windows")

import sounddevice as sd

from .config import AUDIO


def list_devices_windows() -> None:
    """Print disponibili WASAPI devices + raccomandazioni."""
    print("=== Windows Audio Devices ===\n")
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    wasapi_idx = next(
        (i for i, h in enumerate(hostapis) if "wasapi" in h["name"].lower()), None
    )
    if wasapi_idx is None:
        print("⚠️  WASAPI host API non trovato. Su Windows 10/11 dovrebbe essere built-in.")
        return
    print(f"WASAPI host API: idx {wasapi_idx}")
    print()
    print("--- Output devices (utilizzabili in loopback) ---")
    for i, d in enumerate(devices):
        if d["hostapi"] == wasapi_idx and d["max_output_channels"] > 0:
            print(f"  [{i}] {d['name']}  ({d['default_samplerate']} Hz, "
                  f"{d['max_output_channels']} ch)")
    print()
    print("--- Input devices (microfoni) ---")
    for i, d in enumerate(devices):
        if d["hostapi"] == wasapi_idx and d["max_input_channels"] > 0:
            print(f"  [{i}] {d['name']}  ({d['default_samplerate']} Hz, "
                  f"{d['max_input_channels']} ch)")
    print()
    print("Per uso normale: lascia che AUDIO.device_name='' (auto-detect default).")


def _find_default_output() -> int:
    """Trova il default WASAPI output device (per loopback)."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    for i, h in enumerate(hostapis):
        if "wasapi" in h["name"].lower():
            wasapi_idx = i
            break
    else:
        raise RuntimeError("WASAPI host API non disponibile")
    # default output WASAPI
    default_idx = sd.default.device
    if isinstance(default_idx, (list, tuple)):
        default_out = default_idx[1]
    else:
        default_out = default_idx
    # Verifica che sia WASAPI
    if 0 <= default_out < len(devices):
        d = devices[default_out]
        if d["hostapi"] == wasapi_idx and d["max_output_channels"] > 0:
            return default_out
    # Fallback: primo output WASAPI trovato
    for i, d in enumerate(devices):
        if d["hostapi"] == wasapi_idx and d["max_output_channels"] > 0:
            return i
    raise RuntimeError("Nessun output WASAPI trovato")


def _find_default_input() -> int:
    """Trova il default WASAPI input device (microfono)."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    for i, h in enumerate(hostapis):
        if "wasapi" in h["name"].lower():
            wasapi_idx = i
            break
    else:
        raise RuntimeError("WASAPI host API non disponibile")
    default_idx = sd.default.device
    if isinstance(default_idx, (list, tuple)):
        default_in = default_idx[0]
    else:
        default_in = default_idx
    if 0 <= default_in < len(devices):
        d = devices[default_in]
        if d["hostapi"] == wasapi_idx and d["max_input_channels"] > 0:
            return default_in
    for i, d in enumerate(devices):
        if d["hostapi"] == wasapi_idx and d["max_input_channels"] > 0:
            return i
    raise RuntimeError("Nessun input WASAPI trovato")


def audio_stream_loopback(
    sample_rate: int | None = None,
    block_size: int | None = None,
    include_mic: bool = True,
) -> Iterator[np.ndarray]:
    """Stream audio: loopback output (altri partecipanti) + opzionale mic.

    Args:
        sample_rate: Hz (default da config)
        block_size: samples per callback (default da config)
        include_mic: se True, mixa input mic con loopback output

    Yields:
        mono float32 chunks @ sample_rate
    """
    sample_rate = sample_rate or AUDIO.sample_rate
    block_size = block_size or AUDIO.block_size

    out_idx = _find_default_output()
    mic_idx = _find_default_input() if include_mic else None
    print(f"[audio-win] loopback da output #{out_idx}: {sd.query_devices(out_idx)['name']!r}")
    if mic_idx is not None:
        print(f"[audio-win] + mic input #{mic_idx}: {sd.query_devices(mic_idx)['name']!r}")

    # WASAPI loopback settings: sounddevice >= 0.4.6 supporta loopback=True
    try:
        wasapi_settings = sd.WasapiSettings(loopback=True)
    except Exception as e:
        raise RuntimeError(
            f"WASAPI loopback non disponibile in sounddevice ({e}). "
            "Aggiorna sounddevice >= 0.4.6: pip install -U sounddevice"
        )

    audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)

    def loopback_callback(indata, frames, time_, status):
        # Downmix a mono
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = indata.mean(axis=1)
        else:
            mono = indata[:, 0] if indata.ndim == 2 else indata
        try:
            audio_q.put_nowait(mono.copy())
        except queue.Full:
            pass

    # Stream loopback dal default output
    stream = sd.InputStream(
        device=out_idx,
        channels=2,                # most outputs sono stereo
        samplerate=sample_rate,
        blocksize=block_size,
        dtype="float32",
        callback=loopback_callback,
        extra_settings=wasapi_settings,
    )

    # Optional mic stream parallelo (mix manuale nel queue consumer)
    mic_q: queue.Queue[np.ndarray] | None = None
    mic_stream = None
    if mic_idx is not None:
        mic_q = queue.Queue(maxsize=64)
        def mic_callback(indata, frames, time_, status):
            if indata.ndim == 2 and indata.shape[1] > 1:
                mono = indata.mean(axis=1)
            else:
                mono = indata[:, 0] if indata.ndim == 2 else indata
            try:
                mic_q.put_nowait(mono.copy())
            except queue.Full:
                pass
        mic_stream = sd.InputStream(
            device=mic_idx,
            channels=1,
            samplerate=sample_rate,
            blocksize=block_size,
            dtype="float32",
            callback=mic_callback,
        )

    with stream, (mic_stream if mic_stream else _NullContext()):
        print(f"[audio-win] streaming @ {sample_rate} Hz, block={block_size}")
        try:
            while True:
                loopback_chunk = audio_q.get()
                if mic_q is not None:
                    # Mix con mic (se disponibile nel buffer, altrimenti solo loopback)
                    try:
                        mic_chunk = mic_q.get_nowait()
                        # Soft sum, clipping leggero (ognuno 70%)
                        mixed = 0.7 * loopback_chunk + 0.7 * mic_chunk[:len(loopback_chunk)]
                        yield mixed
                    except queue.Empty:
                        yield loopback_chunk
                else:
                    yield loopback_chunk
        except KeyboardInterrupt:
            return


class _NullContext:
    """Context manager no-op per uniformità when no mic_stream."""
    def __enter__(self): return self
    def __exit__(self, *a): return False
