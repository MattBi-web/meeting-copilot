"""Meeting Copilot — entrypoint.

Modalità di lancio:
- `python -m meeting_copilot.main`              → overlay PyQt6 (default)
- `python -m meeting_copilot.main --console`    → modalità console
- `python -m meeting_copilot.main --kb-ui`      → solo Knowledge Manager UI
- `python -m meeting_copilot.main kb <cmd> …`   → CLI knowledge base (es. kb list)

Sottocomandi kb disponibili:
    kb list                       elenca sezioni
    kb create "Nome sezione"      crea nuova
    kb ingest <slug> file1 file2  copia file + reindex
    kb reindex <slug>             rebuild indice
    kb search <slug> "query"      test query
    kb activate <slug>            attiva sezione per call
    kb deactivate <slug>          disattiva
    kb status                     mostra sezioni attive
    kb delete <slug>              elimina sezione

Pipeline runtime:
    audio_capture → VAD → mlx-whisper → (UI overlay | console) + file
    Cmd+Shift+R nell'overlay → retrieval RAG sulle sezioni attive
"""
from __future__ import annotations

# IMPORTANTE: env_loader DEVE essere il primo import,
# legge ~/.config/meeting_copilot/.env e popola os.environ
# prima che config.py o providers leggano le API keys.
# Resiliente: se env_loader.py non sincronizzato → no-op safe.
try:
    from . import env_loader  # noqa: F401
except ImportError:
    pass  # env_loader opzionale, fallback a env vars già impostate (.zshrc)

import argparse
import signal
import sys
import threading
import time
from queue import Empty, Full, Queue

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from . import config
from .audio_capture import audio_stream, audio_stream_with_speaker, list_devices
from .transcript_writer import TranscriptWriter

# Lazy imports per vad/stt: hanno dipendenze pesanti (torch/silero-vad/mlx-whisper)
# che non vogliamo bundlare nel profilo cloud-only (.app friend distribution).
# Quando STT.streaming_enabled=True + backend="deepgram" (profilo standard),
# vad.py e stt.py NON vengono mai chiamati a runtime.
# Importiamo dentro le funzioni che li usano per evitare crash all'avvio bundle.
def _lazy_import_vad():
    from .vad import segmenter, segmenter_with_speaker
    return segmenter, segmenter_with_speaker

def _lazy_import_stt():
    from .stt import transcribe, warmup
    return transcribe, warmup

console = Console()
_stop = threading.Event()

# Politica anti-backlog: la coda fra VAD e STT è piccola (size=2).
# Se STT è in ritardo, VAD scarta segmenti vecchi e mantiene il flusso
# realtime. Meglio perdere 1 frase di 8s fa che mostrare frasi vecchie.
_SEGMENT_QUEUE_MAX = 2


# ─────────────────────────────────────────────────────────────
# Worker threads
# ─────────────────────────────────────────────────────────────
def _handle_sigint(signum, frame):
    _stop.set()


def _vad_worker(segment_q: Queue, device: str | None) -> None:
    """VAD worker con speaker diarization. Yield (segment, speaker_label)."""
    # Lazy import: vad richiede torch+silero-vad, esclusi nel profilo cloud-only.
    _, segmenter_with_speaker = _lazy_import_vad()
    chunks_iter = (
        audio_stream_with_speaker(device_name=device) if device
        else audio_stream_with_speaker()
    )
    dropped = 0
    try:
        for seg, speaker_label in segmenter_with_speaker(chunks_iter):
            if _stop.is_set():
                break
            try:
                segment_q.put_nowait((seg, speaker_label))
            except Full:
                try:
                    segment_q.get_nowait()
                except Empty:
                    pass
                try:
                    segment_q.put_nowait((seg, speaker_label))
                    dropped += 1
                    if dropped % 3 == 1:
                        console.print(
                            f"[yellow][vad] STT in ritardo, droppati {dropped} segmenti vecchi[/]"
                        )
                except Full:
                    pass
    finally:
        try:
            segment_q.put_nowait(None)  # poison pill
        except Full:
            pass


def _stt_worker(segment_q: Queue, on_line) -> None:
    """Consuma (segmento, speaker_label) dalla coda VAD, trascrive, chiama
    `on_line(Transcription, speaker_label)`."""
    # Lazy import: stt richiede mlx-whisper/faster-whisper, esclusi cloud-only
    transcribe, _ = _lazy_import_stt()
    while not _stop.is_set():
        try:
            item = segment_q.get(timeout=0.2)
        except Empty:
            continue
        if item is None:
            break
        segment, speaker_label = item
        t0 = time.monotonic()
        try:
            tr = transcribe(segment)
        except Exception as e:
            console.print(f"[red][stt] error: {e}[/]")
            continue
        stt_elapsed = time.monotonic() - t0
        if tr.text:
            console.print(
                f"[dim][stt] [{speaker_label}] seg={tr.duration_s:.1f}s, "
                f"elapsed={stt_elapsed:.2f}s, "
                f"ratio={stt_elapsed/max(tr.duration_s, 0.01):.2f}x[/]"
            )
            on_line(tr, speaker_label)


def _deepgram_streaming_worker(on_line, device: str | None) -> None:
    """Latency #10: Deepgram WebSocket streaming worker.

    Bypassa VAD locale. Streama audio raw PCM16 al WebSocket Deepgram.
    Per ogni transcript FINAL ricevuto, costruisce Transcription e chiama
    on_line(tr, speaker_label).

    Speaker label: tracking del dominant speaker negli ultimi 1-2s di audio.
    """
    import numpy as np
    from .providers.cloud.deepgram_streaming import DeepgramStreamingSTT
    # Import Transcription dal provider abstraction (provider-agnostic, no ML deps).
    # Equivalente a stt.Transcription ma il modulo providers.base è leggero
    # (importato anche da deepgram_stt + altri cloud providers).
    from .providers.base import Transcription

    # Tracking speaker dominante in finestra mobile (per assegnare label ai transcript)
    _recent_speakers: list[tuple[float, str]] = []  # (timestamp, label)
    _speaker_lock = threading.Lock()

    def _dominant_speaker_recent(window_s: float = 2.0) -> str:
        now = time.monotonic()
        with _speaker_lock:
            recent = [(t, lab) for (t, lab) in _recent_speakers if (now - t) <= window_s]
        if not recent:
            return "unknown"
        # Majority vote
        from collections import Counter
        cnt = Counter(lab for _, lab in recent if lab not in ("unknown", "silence"))
        if not cnt:
            return "unknown"
        return cnt.most_common(1)[0][0]

    def _on_dg_transcript(text: str, is_final: bool, lang: str) -> None:
        if not is_final or not text.strip():
            return
        speaker = _dominant_speaker_recent()
        tr = Transcription(text=text, language=lang, duration_s=0.0)
        console.print(f"[dim][dg-stream] [{speaker}] {text!r}[/]")
        on_line(tr, speaker)

    dg = DeepgramStreamingSTT(on_transcript=_on_dg_transcript)
    dg.start()
    console.print("[cyan][dg-stream] WebSocket Deepgram avviato[/]")

    # Audio stream → PCM16 → feed_audio
    # Mic gain boost: amplifica il segnale se troppo debole (es. audio
    # dall'altro PC via mic ambientale). gain_linear = 10^(dB/20).
    gain_db = getattr(config.STT, "mic_gain_db", 0.0)
    gain_linear = 10.0 ** (gain_db / 20.0) if gain_db != 0 else 1.0
    if gain_db != 0:
        console.print(f"[cyan][dg-stream] mic gain boost: +{gain_db}dB (×{gain_linear:.2f})[/]")

    chunks_iter = (
        audio_stream_with_speaker(device_name=device) if device
        else audio_stream_with_speaker()
    )
    try:
        for mono_chunk, speaker_est in chunks_iter:
            if _stop.is_set():
                break
            # Track speaker per assegnazione label transcript
            label = getattr(speaker_est, "label", "unknown")
            with _speaker_lock:
                _recent_speakers.append((time.monotonic(), label))
                # Cap window
                if len(_recent_speakers) > 200:
                    _recent_speakers[:] = _recent_speakers[-100:]
            # Applica gain boost (se configurato)
            if gain_linear != 1.0:
                mono_chunk = mono_chunk * gain_linear
            # Float32 [-1,1] → int16 PCM bytes (clip per evitare wrap-around)
            pcm16 = np.clip(mono_chunk * 32767.0, -32768, 32767).astype(np.int16).tobytes()
            dg.feed_audio(pcm16)
    finally:
        dg.stop()
        console.print("[dim][dg-stream] stopped[/]")


# ─────────────────────────────────────────────────────────────
# Console mode
# ─────────────────────────────────────────────────────────────
def run_console(device: str | None) -> None:
    signal.signal(signal.SIGINT, _handle_sigint)

    segment_q: Queue = Queue(maxsize=_SEGMENT_QUEUE_MAX)
    line_q: Queue = Queue()

    def on_line(tr, speaker_label):
        line_q.put((tr, speaker_label))

    t_vad = threading.Thread(target=_vad_worker, args=(segment_q, device), daemon=True)
    t_stt = threading.Thread(target=_stt_worker, args=(segment_q, on_line), daemon=True)

    console.print(
        Panel.fit(
            Text.from_markup(
                "[bold cyan]Meeting Copilot[/] — modalità console\n"
                f"Device: [yellow]{device or config.AUDIO.device_name}[/]   "
                f"Model: [yellow]{config.STT.model_size}[/]\n"
                "[dim]Premi Ctrl+C per fermare.[/]"
            ),
            border_style="cyan",
        )
    )

    with TranscriptWriter() as writer:
        console.print(f"[dim]→ scrittura su {writer.path}[/]\n")
        # Pre-warm STT in foreground (skip se non disponibile = cloud-only bundle)
        try:
            _, warmup_fn = _lazy_import_stt()
            warmup_fn()
        except ImportError:
            console.print("[dim][stt] local STT non disponibile (cloud-only build)[/]")
        t_vad.start()
        t_stt.start()
        try:
            while not _stop.is_set():
                try:
                    tr, speaker_label = line_q.get(timeout=0.3)
                except Empty:
                    continue
                lang_tag = f"[dim]\\[{tr.language}][/]" if tr.language else ""
                spk_tag = f"[dim]({speaker_label})[/]" if speaker_label != "unknown" else ""
                console.print(f"{lang_tag} {spk_tag} {tr.text}")
                writer.write(tr.text, language=tr.language, duration_s=tr.duration_s)
        except KeyboardInterrupt:
            _stop.set()
        console.print("\n[dim]Chiusura in corso…[/]")
        t_vad.join(timeout=2)
        t_stt.join(timeout=2)
        console.print(f"[green]✓ Trascrizione salvata: {writer.path}[/]")


# ─────────────────────────────────────────────────────────────
# Overlay mode (default)
# ─────────────────────────────────────────────────────────────
def run_overlay(device: str | None) -> None:
    # Import qui per non pagare il costo di PyQt6 quando si lancia --console
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        console.print(
            "[red]PyQt6 non installato.[/] Esegui: [bold]uv pip install -e '.[ui]'[/]"
        )
        sys.exit(1)
    from .overlay import OverlayWindow, TranscriptionBridge

    app = QApplication(sys.argv)
    app.setApplicationName("Meeting Copilot")
    app.setQuitOnLastWindowClosed(True)

    # macOS: no Dock entry, ma resta visibile (accessory)
    from .macos_window import set_app_accessory_policy
    set_app_accessory_policy()

    # Fase 11: applica user_settings.json se presente
    try:
        from .ui.settings import apply_user_settings_to_config
        apply_user_settings_to_config()
    except Exception as e:
        print(f"[main] settings apply error: {e}")

    # Fase 11: onboarding al primo avvio
    try:
        from .ui.onboarding import run_onboarding_if_needed
        run_onboarding_if_needed()
    except Exception as e:
        print(f"[main] onboarding error: {e}")

    bridge = TranscriptionBridge()
    window = OverlayWindow(bridge)

    # Inizializza telemetry session (Fase 10)
    try:
        from .telemetry import init_session
        init_session(window._session.id)
    except Exception as e:
        print(f"[main] telemetry init error: {e}")

    # Pre-warm STT prima di mostrare la finestra (skip se cloud-only build)
    try:
        _, warmup_fn = _lazy_import_stt()
        warmup_fn()
    except ImportError:
        print("[main] local STT non disponibile (cloud-only build) — skip warmup")

    # Pre-warm RAG reranker (Fase 7) in background, non blocca avvio
    def _warm_reranker():
        try:
            from .kb import reranker as _rr
            _rr.warmup()
        except Exception as e:
            print(f"[main] reranker warmup error: {e}")
    threading.Thread(target=_warm_reranker, daemon=True).start()

    # KB folder watcher: ~/Documents/MeetingCopilot/sections/* → auto-index
    try:
        from .kb.folder_watcher import start_kb_watcher
        _kb_watcher = start_kb_watcher()
    except Exception as e:
        print(f"[main] KB watcher avvio fallito: {e}")
        _kb_watcher = None

    # Apri il transcript writer come context manager scoped al run
    writer = TranscriptWriter()

    def on_line_from_worker(tr, speaker_label):
        # Chiamata DAL thread STT: emit thread-safe via Qt signal
        bridge.new_line.emit(tr.text, tr.language, tr.duration_s, speaker_label)
        # Watcher Agent: notifica nuova riga (in thread Qt via signal nell'overlay)
        try:
            window.on_new_transcript_for_watcher(tr.text, speaker_label)
        except Exception as e:
            print(f"[main] watcher error: {e}")
        # Scrittura file
        writer.write(tr.text, language=tr.language, duration_s=tr.duration_s)

    # Latency #10: scelta del worker pipeline in base a STT config
    _streaming = (
        getattr(config.STT, "streaming_enabled", False)
        and config.STT.backend == "deepgram"
    )
    if _streaming:
        console.print(
            "[cyan][stt] modalità streaming Deepgram WS (bypass VAD locale)[/]"
        )
        t_stream = threading.Thread(
            target=_deepgram_streaming_worker,
            args=(on_line_from_worker, device),
            daemon=True,
        )
        t_stream.start()
        # Placeholder per join finale + compat sotto
        t_vad = t_stream
        t_stt = t_stream
        segment_q = None  # type: ignore[assignment]
    else:
        segment_q = Queue(maxsize=_SEGMENT_QUEUE_MAX)
        t_vad = threading.Thread(target=_vad_worker, args=(segment_q, device), daemon=True)
        t_stt = threading.Thread(target=_stt_worker, args=(segment_q, on_line_from_worker), daemon=True)
        t_vad.start()
        t_stt.start()

    # Mostra la finestra e applica NSWindowSharingNone (avviene in showEvent)
    window.show()
    window.raise_()
    window.activateWindow()

    # Pre-avvia l'orchestrator (warmup LLM in background).
    # Così quando l'utente clicca Ask, i modelli sono già caldi.
    window._ensure_orchestrator()

    # Ctrl+C dal terminale → chiudi pulito
    def on_sigint(*_):
        _stop.set()
        app.quit()

    signal.signal(signal.SIGINT, on_sigint)

    try:
        exit_code = app.exec()
    finally:
        _stop.set()
        # Ferma anche l'orchestrator LLM se è stato avviato
        try:
            if window._orchestrator is not None:
                window._orchestrator.stop()
        except Exception:
            pass
        t_vad.join(timeout=2)
        t_stt.join(timeout=2)
        writer.close()
        console.print(f"[green]✓ Trascrizione salvata: {writer.path}[/]")
        # Fase 5: chiudi session + genera summary post-call
        try:
            summary_path = window.close_session_and_summarize()
            if summary_path:
                console.print(f"[green]✓ Sintesi call salvata: {summary_path}[/]")
        except Exception as e:
            console.print(f"[red]Errore summary: {e}[/]")
        # Fase 15: estrai memorie dalla session e salvale sul scope corrente
        try:
            from .memory import MemoryManager
            from .kb.manager import KnowledgeManager as KM
            active_sections = KM.get_active_sections()
            # Scope di default = prima sezione attiva (se ce ne sono)
            target_scope = active_sections[0] if active_sections else "general"
            n_mem = MemoryManager.save_session_memories(window._session, target_scope)
            if n_mem > 0:
                console.print(
                    f"[green]✓ {n_mem} memorie estratte e salvate nello scope "
                    f"'{target_scope}'.[/]"
                )
        except Exception as e:
            console.print(f"[red]Errore memory extraction: {e}[/]")
        # Fase 10: chiudi telemetry session + ricalcola soglie adaptive
        try:
            from .telemetry import close_session, recompute_thresholds
            close_session(window._session.id)
            recompute_thresholds(reason="session_end")
        except Exception as e:
            console.print(f"[red]Errore telemetry close: {e}[/]")
    sys.exit(exit_code)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def cli() -> None:
    # Dispatch precoce per sottocomandi
    argv = sys.argv[1:]

    # `kb …` → delega a kb/cli.py (typer app)
    if argv and argv[0] == "kb":
        from .kb.cli import app as kb_app
        # Rimuovi "kb" da sys.argv così typer parsa il resto
        sys.argv = [sys.argv[0]] + argv[1:]
        kb_app()
        return

    # `telemetry …` → delega a telemetry/cli.py (typer app)
    if argv and argv[0] == "telemetry":
        from .telemetry.cli import app as tel_app
        sys.argv = [sys.argv[0]] + argv[1:]
        tel_app()
        return

    # `memory …` → delega a memory/cli.py (Fase 15)
    if argv and argv[0] == "memory":
        from .memory.cli import app as mem_app
        sys.argv = [sys.argv[0]] + argv[1:]
        mem_app()
        return

    # `graph …` → delega a kb/graph/cli.py (Fase 17)
    if argv and argv[0] == "graph":
        from .kb.graph.cli import app as graph_app
        sys.argv = [sys.argv[0]] + argv[1:]
        graph_app()
        return

    # `images …` → delega a kb/images_cli.py (Fase 20)
    if argv and argv[0] == "images":
        from .kb.images_cli import app as img_app
        sys.argv = [sys.argv[0]] + argv[1:]
        img_app()
        return

    # `--kb-ui` o `kb-ui` → apri solo Knowledge Manager
    if argv and argv[0] in ("--kb-ui", "kb-ui"):
        from .knowledge_ui import run_kb_ui
        run_kb_ui()
        return

    # Modalità normale: trascrizione (console o overlay)
    p = argparse.ArgumentParser(prog="meeting-copilot")
    p.add_argument("--device", default=None, help="Nome device audio (default: MeetingCapture).")
    p.add_argument("--language", default=None, help="Forza lingua ('it', 'en', ...). Default: auto.")
    p.add_argument("--console", action="store_true", help="Modalità console (no overlay).")
    p.add_argument("--list-devices", action="store_true", help="Elenca device audio ed esci.")
    p.add_argument("--kb-ui", action="store_true", help="Apri solo Knowledge Manager (no trascrizione).")
    args = p.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.kb_ui:
        from .knowledge_ui import run_kb_ui
        run_kb_ui()
        return

    if args.language is not None:
        object.__setattr__(config.STT, "language", args.language)

    if args.console:
        run_console(device=args.device)
    else:
        run_overlay(device=args.device)


if __name__ == "__main__":
    cli()
