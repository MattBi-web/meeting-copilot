"""Tipi STT condivisi — modulo lightweight senza dipendenze ML pesanti.

Mantiene la dataclass Transcription importabile senza pull-in di torch / mlx-whisper /
faster-whisper. Usato da:
- stt.py (provider local, ha tutte le deps)
- providers/cloud/deepgram_stt.py (cloud REST)
- providers/cloud/deepgram_streaming.py (cloud WS)
- main.py (streaming worker)

Pattern: tutti i moduli che producono trascrizioni IMPORTANO da qui, mai da stt.py.
In cloud-only bundle stt.py viene escluso, ma stt_types.py resta disponibile.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Transcription:
    """Risultato di un'iterazione STT."""
    text: str
    language: str
    duration_s: float
