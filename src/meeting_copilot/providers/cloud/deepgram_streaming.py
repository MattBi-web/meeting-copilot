"""Deepgram WebSocket streaming STT — Fase next-level latency.

Pattern alternativo a `deepgram_stt.py` (REST batch).
Mantiene una connessione WebSocket persistente a Deepgram e streama audio
raw in continuo. Risultati arrivano come `interim` (parziali, ~50-150ms)
e `final` (conferma post-utterance).

Architettura:
- Thread #1: produce audio chunks dal microfono (passa attraverso
  `feed_audio()` chiamato dall'audio_capture)
- Thread #2: WebSocket client che riceve transcripts e chiama `on_transcript`
  callback per ogni messaggio

Differenza vs REST batch:
- REST: aspetta segmento completo post-VAD, latency 200-400ms post-end-of-speech
- WebSocket: parole appaiono mentre l'utente parla, latency 50-150ms

NB: questo provider NON implementa l'interfaccia `STTProvider.transcribe(segment)`
perché funziona in modalità streaming push, non batch pull. Va wirato
diversamente nell'audio_capture quando attivo.

Per attivare:
    1. STT.backend = "deepgram-stream"
    2. Audio_capture deve chiamare `feed_audio(pcm16_bytes)` invece di passare
       per VAD/segment buffering.

⚠️ Scaffold sperimentale. Per V0 produzione, usa `deepgram_stt.py` (batch).
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable

from ...config import STT


class DeepgramStreamingSTT:
    """Connessione WebSocket persistente a Deepgram Listen API."""

    def __init__(self, on_transcript: Callable[[str, bool, str], None] | None = None):
        """
        Args:
            on_transcript: callback(text, is_final, language) chiamato per ogni
                           transcript Deepgram (interim e final).
        """
        self.on_transcript = on_transcript or (lambda t, f, l: None)
        self._ws_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ws = None  # websockets.WebSocketClientProtocol
        self._audio_queue: list[bytes] = []
        self._queue_lock = threading.Lock()
        self._connected = False

    def _resolve_api_key(self) -> str:
        key = os.environ.get("DEEPGRAM_API_KEY", "")
        if not key:
            raise RuntimeError("DEEPGRAM_API_KEY non impostata")
        return key

    def _build_ws_url(self) -> str:
        """Costruisce URL WebSocket Deepgram con parametri per streaming."""
        from urllib.parse import urlencode
        # Endpointing + utterance_end_ms configurabili (vedi STT config)
        endpoint_ms = str(getattr(STT, "deepgram_endpointing_ms", 150))
        utterance_ms = str(getattr(STT, "deepgram_utterance_end_ms", 1000))
        params = {
            "model": "nova-3",
            "smart_format": "true",
            "punctuate": "true",
            "interim_results": "true",          # KEY: vuoi parziali (real-time)
            "endpointing": endpoint_ms,         # ms silenzio per end-of-turn (default 150)
            "utterance_end_ms": utterance_ms,   # finalize trigger (default 1000)
            "vad_events": "true",               # eventi VAD per debug audio debole
            "encoding": "linear16",
            "sample_rate": "16000",
            "channels": "1",
        }
        # Multi-language support (Phase Multilingual)
        mode = getattr(STT, "transcription_mode", "single")
        if mode == "multi":
            # Code-switching: Deepgram rileva e trascrive lingue diverse in real time.
            # Supportato da nova-3.
            params["language"] = "multi"
            print(f"[deepgram-stream] language=multi (code-switching)")
        elif mode == "auto":
            # Auto-detect: Deepgram rileva una lingua dominante.
            params["detect_language"] = "true"
            print(f"[deepgram-stream] detect_language=true (auto)")
        elif STT.language:
            # Single lingua forzata (legacy comportamento, più affidabile per call mono-lingua)
            params["language"] = STT.language
        # Keyterm (per nova-3): boost per termini specifici
        try:
            from ...config import PROJECT_ROOT
            vocab_path = PROJECT_ROOT / "session_vocabulary.txt"
            if vocab_path.exists():
                words = vocab_path.read_text(encoding="utf-8").split()
                # Deepgram nova-3 supporta fino a 100 keyterms per request
                keyterms = [w for w in words if 3 <= len(w) <= 30][:100]
                query_pairs = [(k, v) for k, v in params.items()] + [
                    ("keyterm", w) for w in keyterms
                ]
                return f"wss://api.deepgram.com/v1/listen?{urlencode(query_pairs)}"
        except Exception:
            pass
        return f"wss://api.deepgram.com/v1/listen?{urlencode(params)}"

    def start(self) -> None:
        """Avvia il WebSocket thread."""
        if self._ws_thread and self._ws_thread.is_alive():
            return
        self._stop.clear()
        self._ws_thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._ws_thread.start()
        print("[deepgram-stream] WebSocket thread avviato")

    def stop(self) -> None:
        """Ferma il WebSocket cleanly."""
        self._stop.set()
        try:
            if self._ws is not None:
                import asyncio
                asyncio.run(self._ws.close())
        except Exception:
            pass
        if self._ws_thread:
            self._ws_thread.join(timeout=3)

    def feed_audio(self, pcm16_bytes: bytes) -> None:
        """Chiamato dall'audio_capture per pushare audio raw nel WebSocket.

        Args:
            pcm16_bytes: PCM 16-bit signed little-endian mono 16kHz raw bytes.
        """
        with self._queue_lock:
            self._audio_queue.append(pcm16_bytes)

    def _ws_loop(self) -> None:
        """Thread loop: gestisce reconnect + audio send + transcript receive."""
        try:
            import websockets
            import asyncio
        except ImportError as e:
            print(f"[deepgram-stream] websockets non installato: {e}")
            print("[deepgram-stream] pip install websockets")
            return

        url = self._build_ws_url()
        headers = {"Authorization": f"Token {self._resolve_api_key()}"}

        async def runner():
            while not self._stop.is_set():
                try:
                    async with websockets.connect(
                        url, additional_headers=headers, max_size=None,
                    ) as ws:
                        self._ws = ws
                        self._connected = True
                        print("[deepgram-stream] connesso")
                        # Avvia 2 coroutine in parallelo: send audio + recv transcripts
                        await asyncio.gather(
                            self._send_audio_loop(ws),
                            self._recv_transcripts_loop(ws),
                        )
                except websockets.ConnectionClosed:
                    print("[deepgram-stream] connection closed, reconnect in 2s")
                    self._connected = False
                    await asyncio.sleep(2)
                except Exception as e:
                    print(f"[deepgram-stream] error: {e}")
                    self._connected = False
                    await asyncio.sleep(2)

        asyncio.run(runner())

    async def _send_audio_loop(self, ws) -> None:
        """Pulla bytes dalla queue e li manda al WS."""
        import asyncio
        while not self._stop.is_set():
            chunk = None
            with self._queue_lock:
                if self._audio_queue:
                    chunk = self._audio_queue.pop(0)
            if chunk is None:
                await asyncio.sleep(0.02)  # 20ms wait
                continue
            try:
                await ws.send(chunk)
            except Exception as e:
                print(f"[deepgram-stream] send error: {e}")
                break

    async def _recv_transcripts_loop(self, ws) -> None:
        """Riceve JSON messages da Deepgram e invoca callback."""
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            ev_type = msg.get("type")
            if ev_type == "Results":
                channel = msg.get("channel", {})
                alternatives = channel.get("alternatives", [])
                if not alternatives:
                    continue
                text = (alternatives[0].get("transcript") or "").strip()
                is_final = msg.get("is_final", False)
                lang = STT.language or "it"
                if text:
                    try:
                        self.on_transcript(text, is_final, lang)
                    except Exception as e:
                        print(f"[deepgram-stream] callback error: {e}")
