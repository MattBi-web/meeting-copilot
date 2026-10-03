"""Utilities macOS-specific per l'overlay.

Il punto chiave è `make_invisible_to_screen_share`: applica a una QWidget
l'attributo NSWindow `sharingType = NSWindowSharingNone`, che rende la finestra
invisibile a TUTTE le API di screen capture di macOS:
- "Share Screen" di Meet/Teams/Zoom/Webex
- Screenshot (Cmd+Shift+3/4)
- Registrazione QuickTime
- AirPlay / Sidecar mirroring
- App di registrazione di terze parti (OBS, Loom, ecc.)

Funziona dal lato sistema operativo: macOS sa che quella finestra è "privata"
e la esclude dal buffer di cattura. Non è un trucco visivo, è una proprietà
nativa supportata da Apple dal 2010 (10.5+).
"""
from __future__ import annotations

import sys


# Costante NSWindow.SharingType:
#   NSWindowSharingNone = 0  → finestra esclusa da screen capture
#   NSWindowSharingReadOnly = 1 (default) → catturabile
#   NSWindowSharingReadWrite = 2 → catturabile + condivisa
_NSWindowSharingNone = 0


def make_invisible_to_screen_share(qwidget) -> bool:
    """Applica NSWindowSharingNone alla NSWindow associata alla QWidget.

    Args:
        qwidget: una QWidget PyQt6 che è già stata mostrata (`show()` chiamato).
                 Il chiamante DEVE invocare questa funzione DOPO show(),
                 perché winId() ritorna 0 finché la finestra non è realizzata.

    Returns:
        True se l'operazione è andata a buon fine, False altrimenti.
        Su non-macOS ritorna False senza errore.
    """
    if sys.platform != "darwin":
        return False

    try:
        # Import dentro la funzione: PyObjC è una dipendenza pesante,
        # importarlo solo quando serve velocizza l'avvio.
        import objc
        from AppKit import NSView  # noqa: F401  (registra il bridge AppKit)
    except ImportError as e:
        print(
            f"[macos_window] PyObjC non disponibile: {e}. "
            "Installa con: uv pip install -e '.[ui]'",
            file=sys.stderr,
        )
        return False

    try:
        # winId() su macOS ritorna un puntatore al NSView nativo
        win_id = int(qwidget.winId())
        if win_id == 0:
            print(
                "[macos_window] winId()==0: chiama make_invisible_to_screen_share "
                "DOPO qwidget.show().",
                file=sys.stderr,
            )
            return False

        # Wrappa il puntatore in un oggetto PyObjC NSView
        ns_view = objc.objc_object(c_void_p=win_id)

        # NSView ha .window() che ritorna l'NSWindow contenitore
        ns_window = ns_view.window()
        if ns_window is None:
            print("[macos_window] NSWindow non trovata.", file=sys.stderr)
            return False

        # Applica il flag di esclusione da screen capture
        ns_window.setSharingType_(_NSWindowSharingNone)
        return True

    except Exception as e:  # pragma: no cover
        print(f"[macos_window] errore applicando NSWindowSharingNone: {e}", file=sys.stderr)
        return False


def set_app_accessory_policy() -> bool:
    """Imposta l'app come 'Accessory': resta visibile sullo schermo, ma
    NON appare nel Dock né nel menu bar. Equivalente a `LSUIElement=true`
    nel plist, ma applicato a runtime.

    Da chiamare DOPO QApplication() ma PRIMA o DOPO show() della prima finestra.
    """
    if sys.platform != "darwin":
        return False
    try:
        from AppKit import NSApp
        # NSApplicationActivationPolicyAccessory = 1
        # NSApplicationActivationPolicyRegular = 0
        # NSApplicationActivationPolicyProhibited = 2
        NSApp.setActivationPolicy_(1)
        return True
    except Exception as e:
        print(f"[macos_window] errore setActivationPolicy: {e}", file=sys.stderr)
        return False


def set_window_always_on_top_floating(qwidget) -> bool:
    """Imposta un livello di finestra alto (sopra alle app normali ma sotto
    a alert di sistema). Usato in combinazione con il flag Qt
    WindowStaysOnTopHint per rendere la finestra "saldamente flottante".

    Funziona meglio del solo flag Qt quando si interagisce con apps fullscreen
    (es. Meet/Zoom in modalità presentazione)."""
    if sys.platform != "darwin":
        return False
    try:
        import objc
        from AppKit import NSFloatingWindowLevel

        win_id = int(qwidget.winId())
        if win_id == 0:
            return False
        ns_view = objc.objc_object(c_void_p=win_id)
        ns_window = ns_view.window()
        if ns_window is None:
            return False
        ns_window.setLevel_(NSFloatingWindowLevel)
        return True
    except Exception as e:
        print(f"[macos_window] errore impostando window level: {e}", file=sys.stderr)
        return False
