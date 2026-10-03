"""Memory cross-call — Fase 15.

Sistema di memoria episodica/semantica che persiste tra una call e l'altra.
- Estrae fatti rilevanti dalla session a fine call (Extractor)
- Salva in SQLite + embedding per retrieval semantico
- Cerca memorie rilevanti durante una nuova call e le aggiunge al context RAG

Differenza vs KB statica:
- KB = documenti fissi (PPT, FAQ, scripts) che NON cambiano
- Memory = facts dinamici che CRESCONO con ogni call

Esempi di memoria:
- "Cliente Pharma X ha 500 dipendenti e usa Salesforce"
- "Durante la call del 15/5 hanno chiesto demo per fine maggio"
- "Decision maker: Carlo Rossi (CTO), Maria Bianchi (CFO)"
- "Obiezione ricorrente: latenza percepita > 2s inaccettabile"

Files persistenti:
- ~/.meeting_copilot/memory.db          (SQLite memorie + embeddings)
"""
from .manager import MemoryManager
from .store import MemoryRecord

__all__ = ["MemoryManager", "MemoryRecord"]
