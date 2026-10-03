"""Global hotkeys via PyObjC NSEvent monitor.

A differenza delle QShortcut PyQt6 (che funzionano solo con focus sull'app),
qui usiamo `NSEvent.addGlobalMonitorForEventsMatchingMask_handler_` che
intercetta keypress a livello sistema: funziona anche quando Meet, Zoom, Teams
sono in foreground.

⚠️ Richiede permission **Accessibility** in System Settings → Privacy & Security.
Al primo lancio macOS popperà una richiesta di permesso.

Hotkey gestite (modificabili in config futura):
- Cmd+Shift+R → Ask (trigger Analyzer+Composer)
- Cmd+Shift+S → Save passage to KB
- Cmd+Shift+N → Focus notes panel
- Cmd+Shift+T → Toggle visibility overlay
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class HotkeyBinding:
    key: str               # "R", "S", "N", "T" (single uppercase char)
    modifiers: int         # bit flags
    callback: Callable[[], None]


class GlobalHotkeys:
    """Wrapper su NSEvent global monitor. Si tiene il monitor handle vivo
    per non perdere gli eventi (PyObjC garbage collection)."""

    def __init__(self):
        self._monitor = None
        self._local_monitor = None
        self._bindings: list[HotkeyBinding] = []
        self._enabled = False

    def register(self, key: str, callback: Callable[[], None]) -> None:
        """Registra una hotkey Cmd+Shift+<key>."""
        # NSEventModifierFlagCommand = 1 << 20 = 0x100000
        # NSEventModifierFlagShift   = 1 << 17 = 0x20000
        # Mask: combinato + altri flag puliti
        cmd_shift = (1 << 20) | (1 << 17)
        self._bindings.append(HotkeyBinding(
            key=key.upper(),
            modifiers=cmd_shift,
            callback=callback,
        ))

    def start(self) -> bool:
        if sys.platform != "darwin":
            return False
        if self._enabled:
            return True
        try:
            from AppKit import NSEvent
        except ImportError as e:
            print(f"[hotkeys] PyObjC AppKit non disponibile: {e}", file=sys.stderr)
            return False

        # NSEventMaskKeyDown = 1 << 10
        key_down_mask = 1 << 10

        def handler(event):
            try:
                chars = (event.charactersIgnoringModifiers() or "").upper()
                flags = int(event.modifierFlags())
                # Filtra solo i bit di interesse (Cmd, Shift, Ctrl, Alt)
                relevant = flags & ((1 << 20) | (1 << 17) | (1 << 18) | (1 << 19))
                for b in self._bindings:
                    if chars == b.key and (relevant & b.modifiers) == b.modifiers:
                        try:
                            b.callback()
                        except Exception as e:
                            print(f"[hotkeys] callback error: {e}", file=sys.stderr)
                        return  # eat the event
            except Exception as e:
                print(f"[hotkeys] handler error: {e}", file=sys.stderr)
            return event   # passa l'evento al sistema

        # Global monitor: cattura eventi quando l'app NON ha focus
        self._monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            key_down_mask, handler
        )
        # Local monitor: cattura eventi quando l'app HA focus (per coerenza)
        self._local_monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
            key_down_mask, handler
        )
        self._enabled = True
        print("[hotkeys] ✓ Global hotkeys attive: "
              + ", ".join(f"Cmd+Shift+{b.key}" for b in self._bindings))
        return True

    def stop(self) -> None:
        if sys.platform != "darwin":
            return
        try:
            from AppKit import NSEvent
            if self._monitor is not None:
                NSEvent.removeMonitor_(self._monitor)
                self._monitor = None
            if self._local_monitor is not None:
                NSEvent.removeMonitor_(self._local_monitor)
                self._local_monitor = None
        except Exception:
            pass
        self._enabled = False
