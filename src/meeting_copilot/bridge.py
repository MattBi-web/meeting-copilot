"""Local Bridge — mini FastAPI server sul Mac per controllo remoto dell'overlay.

Pattern: gira come daemon background sul Mac dell'utente (`python -m meeting_copilot.bridge`),
espone endpoint HTTP localhost:9999. La dashboard web (anche se gira sulla VM) può
fare fetch a localhost dal browser dell'utente — funziona perché il browser è sul Mac.

Endpoint:
    GET  /bridge/ping          → {"alive": true, "version": "..."}
    POST /bridge/overlay/start → subprocess `python -m meeting_copilot.main`
    POST /bridge/overlay/stop  → kill del subprocess corrente
    GET  /bridge/overlay/status → {"running": bool, "pid": int|None, "started_at": float}

CORS aperto a tutto (è un servizio locale, accessed solo dal tuo browser).
NO auth (è localhost-only, listen su 127.0.0.1).

Avvio:
    python -m meeting_copilot.bridge

Per auto-start al login Mac (opzionale, futuro):
    LaunchAgent in ~/Library/LaunchAgents/com.meetingcopilot.bridge.plist
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as e:
    raise RuntimeError(
        "FastAPI non installato. pip install fastapi 'uvicorn[standard]'"
    ) from e


_BRIDGE_VERSION = "0.1.0"
_overlay_proc: subprocess.Popen | None = None
_overlay_started_at: float = 0.0


def _detect_project_root() -> Path:
    """Trova la root del progetto (dove c'è pyproject.toml + src/meeting_copilot)."""
    # __file__ è src/meeting_copilot/bridge.py → root = parents[2]
    return Path(__file__).resolve().parents[2]


def _python_executable() -> str:
    """Python da usare: preferisci venv se attivo, altrimenti sys.executable."""
    root = _detect_project_root()
    venv_python = root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _start_overlay() -> tuple[bool, str]:
    """Spawna overlay come subprocess background."""
    global _overlay_proc, _overlay_started_at
    if _overlay_proc and _overlay_proc.poll() is None:
        return False, f"Overlay già in esecuzione (PID {_overlay_proc.pid})"
    root = _detect_project_root()
    py = _python_executable()
    try:
        # Eredita env corrente (incluse env var come ANTHROPIC_API_KEY, DEEPGRAM_API_KEY)
        proc = subprocess.Popen(
            [py, "-m", "meeting_copilot.main"],
            cwd=str(root),
            env=os.environ.copy(),
            # Stdout/stderr verso pipe → vediamo log se vuoi
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # Detach da bridge: se il bridge muore, overlay vive
            start_new_session=True,
        )
        _overlay_proc = proc
        _overlay_started_at = time.time()
        return True, f"Overlay avviato (PID {proc.pid})"
    except Exception as e:
        return False, f"Errore start: {e}"


def _stop_overlay() -> tuple[bool, str]:
    """Manda SIGTERM all'overlay corrente."""
    global _overlay_proc
    if not _overlay_proc or _overlay_proc.poll() is not None:
        return False, "Nessun overlay in esecuzione"
    pid = _overlay_proc.pid
    try:
        # SIGTERM al process group (include figli)
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        # Aspetta max 3s per shutdown gentile
        try:
            _overlay_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            _overlay_proc.wait(timeout=1)
        _overlay_proc = None
        return True, f"Overlay PID {pid} terminato"
    except Exception as e:
        return False, f"Errore stop: {e}"


def _overlay_status() -> dict:
    if _overlay_proc and _overlay_proc.poll() is None:
        return {
            "running": True,
            "pid": _overlay_proc.pid,
            "started_at": _overlay_started_at,
            "uptime_s": time.time() - _overlay_started_at,
        }
    return {"running": False, "pid": None, "started_at": None, "uptime_s": 0}


# ──────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────
def build_bridge_app() -> "FastAPI":
    app = FastAPI(title="Meeting Copilot Local Bridge", version=_BRIDGE_VERSION)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/bridge/ping")
    def ping() -> dict:
        return {
            "alive": True,
            "version": _BRIDGE_VERSION,
            "project_root": str(_detect_project_root()),
            "python": _python_executable(),
        }

    @app.get("/bridge/env-check")
    def env_check() -> dict:
        """Verifica env var critiche presenti sul Mac dell'utente."""
        return {
            "deepgram_key_present": bool(os.environ.get("DEEPGRAM_API_KEY", "")),
            "anthropic_key_present": bool(os.environ.get("ANTHROPIC_API_KEY", "")),
            "server_url": os.environ.get("MEETING_COPILOT_SERVER_URL", ""),
            "server_key_present": bool(os.environ.get("MEETING_COPILOT_API_KEY", "")),
        }

    @app.post("/bridge/overlay/start")
    def overlay_start() -> dict:
        ok, msg = _start_overlay()
        return {"ok": ok, "message": msg, **_overlay_status()}

    @app.post("/bridge/overlay/stop")
    def overlay_stop() -> dict:
        ok, msg = _stop_overlay()
        return {"ok": ok, "message": msg}

    @app.get("/bridge/overlay/status")
    def overlay_status() -> dict:
        return _overlay_status()

    return app


def run(host: str = "127.0.0.1", port: int = 9999) -> None:
    try:
        import uvicorn
    except ImportError as e:
        raise RuntimeError("uvicorn non installato") from e
    print(f"[bridge] starting on {host}:{port} (Mac-local control plane)")
    print(f"[bridge] project root: {_detect_project_root()}")
    print(f"[bridge] python: {_python_executable()}")
    uvicorn.run(build_bridge_app(), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run()
