"""Path resolution — gestisce modalità dev (codice in repo) vs bundled (.app).

Quando l'app gira come .app PyInstaller bundle:
- Codice + dipendenze sono dentro MeetingCopilot.app/Contents/Resources/
- File scrivibili (transcripts, sessions, kb data) NON possono stare nel bundle
  (sandbox + permissions). Vanno in ~/Library/Application Support/MeetingCopilot/
- Config user-editabili (.env API keys): ~/.config/meeting_copilot/

Quando gira in dev mode (python -m meeting_copilot.main):
- PROJECT_ROOT = repo locale
- transcripts/sessions/kb in repo (per debug rapido)

Questo modulo è la SINGLE SOURCE OF TRUTH per i paths. Tutti gli altri moduli
dovrebbero importare da qui invece di calcolarli inline.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def is_bundled() -> bool:
    """True se gira come PyInstaller .app bundle (frozen)."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def bundle_resources_dir() -> Path | None:
    """Path delle Resources del bundle (read-only) se bundled, None altrimenti.

    Contiene: session_vocabulary.txt, sample KB (kb/acme_demo/), ecc.
    """
    if not is_bundled():
        return None
    # sys._MEIPASS è la directory temporanea PyInstaller, contiene tutti
    # i datas inclusi nello spec.
    return Path(sys._MEIPASS)


def _platform_data_dir() -> Path:
    """Standard OS-specific user data directory.

    macOS:    ~/Library/Application Support/MeetingCopilot
    Windows:  %APPDATA%/MeetingCopilot (typically ~/AppData/Roaming/MeetingCopilot)
    Linux:    $XDG_DATA_HOME/MeetingCopilot o ~/.local/share/MeetingCopilot
    """
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "MeetingCopilot"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "MeetingCopilot"
        return Path.home() / "AppData" / "Roaming" / "MeetingCopilot"
    # Linux + altri
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "MeetingCopilot"
    return Path.home() / ".local" / "share" / "MeetingCopilot"


def _platform_config_dir() -> Path:
    """Standard OS-specific user config directory.

    macOS / Linux: ~/.config/meeting_copilot
    Windows:       %APPDATA%/MeetingCopilot/config
    """
    if sys.platform == "win32":
        return _platform_data_dir() / "config"
    return Path.home() / ".config" / "meeting_copilot"


def user_data_dir() -> Path:
    """Dir scrivibile per dati utente (transcripts, sessions, KB).

    Bundled mode → standard OS app-data dir
    Dev mode → repo root (compatibilità storica)
    """
    if is_bundled():
        d = _platform_data_dir()
        d.mkdir(parents=True, exist_ok=True)
        return d
    # Dev mode: repo root (parents[2] da src/meeting_copilot/paths.py)
    return Path(__file__).resolve().parents[2]


def user_config_dir() -> Path:
    """Dir per config user-editabili (.env, settings JSON).

    Cross-platform via _platform_config_dir.
    """
    d = _platform_config_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_kb_drop_folder() -> Path:
    """Cartella user-visible per drop KB documents.

    Sempre ~/Documents/MeetingCopilot/sections/ — facile da trovare in Finder.
    """
    d = Path.home() / "Documents" / "MeetingCopilot" / "sections"
    d.mkdir(parents=True, exist_ok=True)
    return d


def transcripts_dir() -> Path:
    """Dir per transcript files."""
    d = user_data_dir() / "transcripts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sessions_dir() -> Path:
    """Dir base per session JSON + summary."""
    d = user_data_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def kb_data_dir() -> Path:
    """Dir per KB chunks/index (ChromaDB local store)."""
    d = user_data_dir() / "kb"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_bundled_sample_kb() -> Path | None:
    """Sample KB section inclusa nel bundle (acme_demo).

    Bundled mode: ritorna percorso dentro Resources/.
    Dev mode: ritorna repo/kb/acme_demo se esiste, altrimenti None.

    Usato dall'onboarding wizard per offrire "Inizia con sample KB".
    """
    if is_bundled():
        res = bundle_resources_dir()
        if res:
            p = res / "kb" / "acme_demo"
            return p if p.exists() else None
    repo_root = Path(__file__).resolve().parents[2]
    p = repo_root / "kb" / "acme_demo"
    return p if p.exists() else None


def runtime_summary() -> dict:
    """Helper debug: stampa tutti i path risolti (per logging avvio)."""
    return {
        "is_bundled": is_bundled(),
        "bundle_resources": str(bundle_resources_dir()) if is_bundled() else None,
        "user_data_dir": str(user_data_dir()),
        "user_config_dir": str(user_config_dir()),
        "user_kb_drop": str(user_kb_drop_folder()),
        "transcripts_dir": str(transcripts_dir()),
        "sessions_dir": str(sessions_dir()),
        "kb_data_dir": str(kb_data_dir()),
        "platform": sys.platform,
        "python": sys.version.split()[0],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(runtime_summary(), indent=2))
