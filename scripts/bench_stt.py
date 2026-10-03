#!/usr/bin/env python3
"""Benchmark STT streaming: Deepgram Nova-3 vs AssemblyAI Universal-Streaming.

Risponde a OQ-17 (DESIGN_THIN_CLIENT.md §11): quale vendor per il thin client,
su qualità italiano/inglese e latenza. Da eseguire sulla macchina di Matteo
(servono le chiavi e la rete).

Uso:
    export DEEPGRAM_API_KEY=...
    export ASSEMBLYAI_API_KEY=...
    python scripts/bench_stt.py --wav call_it.wav --ref call_it.txt
    python scripts/bench_stt.py --wav call_en.wav --ref call_en.txt --language en
    python scripts/bench_stt.py --wav call_it.wav --vendor deepgram   # uno solo

Output per vendor: transcript completo, latenza primo final, WER vs --ref
(se fornito: file di testo con la trascrizione di riferimento).

Requisiti WAV: PCM16 mono 16kHz (ffmpeg -i in -ar 16000 -ac 1 -sample_fmt s16 out.wav).
Costi indicativi/h (2026-07): Deepgram ~$0.46 · AssemblyAI ~$0.15.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import wave
from urllib.parse import urlencode

SAMPLE_RATE = 16_000
FRAME_MS = 100
FRAME_BYTES = SAMPLE_RATE * 2 * FRAME_MS // 1000


def load_pcm(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            sys.exit(f"{path}: serve PCM16 mono 16kHz "
                     f"(ffmpeg -i {path} -ar 16000 -ac 1 -sample_fmt s16 out.wav)")
        return w.readframes(w.getnframes())


def wer(ref: str, hyp: str) -> float:
    """Word Error Rate via Levenshtein parole (case/punteggiatura-insensitive)."""
    import re
    norm = lambda s: re.sub(r"[^\w\s]", "", s.lower()).split()
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / len(r)


async def stream_audio(ws, pcm: bytes, speed: float) -> None:
    delay = (FRAME_MS / 1000.0) / speed
    for i in range(0, len(pcm), FRAME_BYTES):
        await ws.send(pcm[i:i + FRAME_BYTES])
        await asyncio.sleep(delay)


async def bench_deepgram(pcm: bytes, language: str, speed: float) -> dict:
    import websockets
    key = os.environ.get("DEEPGRAM_API_KEY") or sys.exit("DEEPGRAM_API_KEY mancante")
    params = {
        "model": "nova-3", "smart_format": "true", "punctuate": "true",
        "interim_results": "true", "endpointing": "150", "utterance_end_ms": "1000",
        "encoding": "linear16", "sample_rate": str(SAMPLE_RATE), "channels": "1",
        "language": language,
    }
    url = f"wss://api.deepgram.com/v1/listen?{urlencode(params)}"
    finals, t0, first_final = [], time.monotonic(), None

    async with websockets.connect(
        url, additional_headers={"Authorization": f"Token {key}"}, max_size=None,
    ) as ws:
        async def recv():
            nonlocal first_final
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") != "Results":
                    continue
                alts = msg.get("channel", {}).get("alternatives", [])
                text = (alts[0].get("transcript") or "").strip() if alts else ""
                if text and msg.get("is_final"):
                    if first_final is None:
                        first_final = time.monotonic() - t0
                    finals.append(text)

        recv_task = asyncio.create_task(recv())
        await stream_audio(ws, pcm, speed)
        await ws.send(json.dumps({"type": "CloseStream"}))
        try:
            await asyncio.wait_for(recv_task, timeout=10)
        except asyncio.TimeoutError:
            recv_task.cancel()

    return {"vendor": "deepgram nova-3", "transcript": " ".join(finals),
            "first_final_s": first_final}


async def bench_assemblyai(pcm: bytes, language: str, speed: float) -> dict:
    import websockets
    key = os.environ.get("ASSEMBLYAI_API_KEY") or sys.exit("ASSEMBLYAI_API_KEY mancante")
    # Universal-Streaming v3. NB: verifica supporto lingua (l'italiano su
    # streaming è arrivato dopo l'inglese — se rifiuta, il benchmark IT decade
    # su Deepgram automaticamente e questo è già un dato per OQ-17).
    params = {"sample_rate": str(SAMPLE_RATE), "encoding": "pcm_s16le"}
    if language and language != "en":
        params["language"] = language
    url = f"wss://streaming.assemblyai.com/v3/ws?{urlencode(params)}"
    finals, t0, first_final = [], time.monotonic(), None

    async with websockets.connect(
        url, additional_headers={"Authorization": key}, max_size=None,
    ) as ws:
        async def recv():
            nonlocal first_final
            async for raw in ws:
                msg = json.loads(raw)
                # v3: type=Turn con end_of_turn=true e campo transcript
                if msg.get("type") == "Turn" and msg.get("end_of_turn"):
                    text = (msg.get("transcript") or "").strip()
                    if text:
                        if first_final is None:
                            first_final = time.monotonic() - t0
                        finals.append(text)

        recv_task = asyncio.create_task(recv())
        await stream_audio(ws, pcm, speed)
        await ws.send(json.dumps({"type": "Terminate"}))
        try:
            await asyncio.wait_for(recv_task, timeout=10)
        except asyncio.TimeoutError:
            recv_task.cancel()

    return {"vendor": "assemblyai universal-streaming", "transcript": " ".join(finals),
            "first_final_s": first_final}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--ref", help="file .txt con trascrizione di riferimento (per WER)")
    ap.add_argument("--language", default="it")
    ap.add_argument("--vendor", choices=["deepgram", "assemblyai", "both"], default="both")
    ap.add_argument("--speed", type=float, default=2.0,
                    help="streaming speed (2x default: benchmark, non latenza pura)")
    args = ap.parse_args()

    pcm = load_pcm(args.wav)
    ref = open(args.ref).read() if args.ref else None
    duration = len(pcm) / (SAMPLE_RATE * 2)
    print(f"Audio: {args.wav} · {duration:.1f}s · lingua={args.language} · speed={args.speed}x\n")

    results = []
    if args.vendor in ("deepgram", "both"):
        results.append(await bench_deepgram(pcm, args.language, args.speed))
    if args.vendor in ("assemblyai", "both"):
        try:
            results.append(await bench_assemblyai(pcm, args.language, args.speed))
        except Exception as e:
            print(f"⚠️ AssemblyAI fallito ({e}) — dato utile per OQ-17 se lingua={args.language}")

    for r in results:
        print(f"── {r['vendor']} ──")
        print(f"  primo final: {r['first_final_s']:.2f}s" if r["first_final_s"]
              else "  primo final: n/a")
        if ref:
            print(f"  WER vs ref: {wer(ref, r['transcript']) * 100:.1f}%")
        print(f"  transcript: {r['transcript'][:500]}"
              f"{'…' if len(r['transcript']) > 500 else ''}\n")

    if ref and len(results) == 2:
        w0, w1 = (wer(ref, r["transcript"]) for r in results)
        better = results[0 if w0 <= w1 else 1]["vendor"]
        print(f"→ WER migliore: {better} ({min(w0, w1) * 100:.1f}% vs {max(w0, w1) * 100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
