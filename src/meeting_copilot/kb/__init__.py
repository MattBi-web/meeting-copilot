"""Knowledge Base — gestione sezioni, ingest, retrieval.

Layout filesystem:
    ~/Dev/meeting_copilot/kb/
    └── <section_slug>/
        ├── sources/            ← file originali (drag&drop o copy)
        │   ├── slide.pdf
        │   ├── pricing.xlsx
        │   ├── _urls.txt        ← un URL per riga, viene fetchato all'ingest
        │   └── ...
        ├── chunks.jsonl        ← chunks estratti (rebuilduibili)
        └── index/              ← ChromaDB persistent collection

Tre interfacce parallele sullo stesso filesystem:
- CLI: `meeting-copilot kb create|list|ingest|search|reindex`
- UI Knowledge Manager: finestra PyQt6 dedicata
- Filesystem diretto: copi/sposti file, lanci reindex

Stato attivo (quali sezioni sono "on" per la call corrente):
    ~/.config/meeting_copilot/active_kb.json
"""
from .manager import (
    KnowledgeManager,
    Section,
    Chunk,
    SearchHit,
)

__all__ = ["KnowledgeManager", "Section", "Chunk", "SearchHit"]
