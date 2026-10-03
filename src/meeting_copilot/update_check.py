"""Update check leggero — M4.

Niente Sparkle/framework nativi: all'avvio l'app scarica (best-effort, in
background) un manifest JSON e confronta la versione. Se c'è una release
nuova, mostra un messaggio non invasivo con il link.

Manifest atteso (hostato su landing site o GitHub releases):
    {"version": "0.3.0", "url": "https://.../MeetingCopilot-0.3.0.dmg",
     "notes": "…"}

URL del manifest: env MC_UPDATE_MANIFEST_URL, altrimenti il default sotto
(placeholder finché il dominio non esiste — con {{BRAND}} pendente).
Se l'URL non risponde: silenzio, mai bloccare l'avvio.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.request

DEFAULT_MANIFEST_URL = ""  # TODO post-brand: https://<dominio>/mc/update.json


def _parse_version(v: str) -> tuple:
    parts = []
    for piece in v.split("-")[0].split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def check_async(current_version: str, notify) -> None:
    """notify(str) chiamata SOLO se esiste una versione più nuova.

    NB: notify arriva da un thread di background — in Qt passare per un
    segnale se si tocca la UI (thin_app usa statusBar già thread-safe-ish
    via showMessage; per messaggi più ricchi usare pyqtSignal).
    """
    url = os.environ.get("MC_UPDATE_MANIFEST_URL", DEFAULT_MANIFEST_URL)
    if not url:
        return

    def _worker() -> None:
        try:
            with urllib.request.urlopen(url, timeout=6) as r:
                manifest = json.loads(r.read().decode("utf-8"))
            latest = str(manifest.get("version", ""))
            if latest and _parse_version(latest) > _parse_version(current_version):
                notify(f"Nuova versione {latest} disponibile — "
                       f"{manifest.get('url', '')}")
        except Exception:
            pass  # mai disturbare l'avvio per un update check

    threading.Thread(target=_worker, daemon=True).start()
