"""KB folder watcher — drag-drop UX per knowledge base.

Pattern:
- Cartella user-visible: ~/Documents/MeetingCopilot/sections/<nome_sezione>/
- Trascini un file dentro → watcher rileva → KnowledgeManager.add + reindex auto
- Sottocartelle = sezioni (auto-create se non esistono)
- Le sezioni gestite via folder appaiono nella dashboard normalmente

Vantaggi rispetto a CLI:
- Non-tech user lavora come con Dropbox: drop & forget
- Ogni cartella = sezione, organizzazione spontanea
- Auto-sync se editi/rimuovi file

Stato:
- Avviato come thread daemon all'avvio dell'overlay (vedi main.py)
- Configurabile via STT.kb_drop_folder (default ~/Documents/MeetingCopilot/sections)
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from .manager import KnowledgeManager


DEFAULT_DROP_FOLDER = Path.home() / "Documents" / "MeetingCopilot" / "sections"


def _slugify_folder_name(name: str) -> str:
    """Trasforma il nome cartella in slug compatibile KnowledgeManager."""
    import re
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "default"


class KBFolderWatcher:
    """Watcher su ~/Documents/MeetingCopilot/sections/ per drop-and-index UX.

    Polling-based (non usa watchdog/inotify per evitare dep extra).
    Scan ogni N secondi: rileva file nuovi/modificati/rimossi → sync con KB.
    """

    def __init__(
        self,
        root: Path = DEFAULT_DROP_FOLDER,
        poll_interval_s: float = 5.0,
    ):
        self.root = root
        self.poll_interval_s = poll_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Snapshot {file_path: mtime} per sezione, per detection cambiamenti
        self._snapshot: dict[str, dict[str, float]] = {}

    def start(self) -> None:
        """Avvia watcher come thread daemon."""
        if self._thread and self._thread.is_alive():
            return
        # Crea root se non esiste + README esempio
        self._ensure_root()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="KBFolderWatcher")
        self._thread.start()
        print(f"[kb-watcher] avviato su {self.root}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _ensure_root(self) -> None:
        """Crea root folder + README spiegativo."""
        self.root.mkdir(parents=True, exist_ok=True)
        readme = self.root / "README.txt"
        if not readme.exists():
            readme.write_text(
                "Meeting Copilot — Knowledge Base drop folder\n"
                "═══════════════════════════════════════════\n\n"
                "Crea una sottocartella per ogni 'sezione' KB. Es:\n\n"
                "  sections/\n"
                "  ├── pitch_producta/\n"
                "  │   ├── overview.pdf\n"
                "  │   ├── pricing.md\n"
                "  │   └── faq.docx\n"
                "  ├── cliente_pharma/\n"
                "  │   └── …\n"
                "  └── obiezioni_tipiche/\n"
                "      └── …\n\n"
                "Quando trascini un file dentro, viene indicizzato AUTOMATICAMENTE\n"
                "(entro 5-10 secondi). Puoi cancellare/modificare i file: il sync\n"
                "è bidirezionale.\n\n"
                "Per attivare una sezione durante la call: usa la dashboard o il\n"
                "menu Knowledge dell'overlay.\n\n"
                "Formati supportati: .pdf .docx .pptx .md .txt .html .url\n",
                encoding="utf-8",
            )

    def _loop(self) -> None:
        """Loop principale: polling cartelle, sync con KB."""
        while not self._stop.is_set():
            try:
                self._scan_and_sync()
            except Exception as e:
                print(f"[kb-watcher] errore scan: {e}")
            # Interruttibile (no time.sleep diretto)
            for _ in range(int(self.poll_interval_s * 2)):
                if self._stop.is_set():
                    return
                time.sleep(0.5)

    def _scan_and_sync(self) -> None:
        """Scansiona root, rileva diff con snapshot, propaga a KnowledgeManager."""
        if not self.root.exists():
            return
        # Per ogni sottocartella = sezione
        section_dirs = [d for d in self.root.iterdir() if d.is_dir()]
        for sec_dir in section_dirs:
            slug = _slugify_folder_name(sec_dir.name)
            self._sync_section(slug, sec_dir)

    def _sync_section(self, slug: str, sec_dir: Path) -> None:
        """Sync di una singola sezione."""
        # Scan file rilevanti (skip hidden/temp)
        valid_ext = {".pdf", ".docx", ".pptx", ".md", ".txt", ".html", ".url"}
        current_files: dict[str, float] = {}
        for f in sec_dir.rglob("*"):
            if not f.is_file():
                continue
            if f.name.startswith("."):
                continue
            if f.suffix.lower() not in valid_ext:
                continue
            try:
                current_files[str(f)] = f.stat().st_mtime
            except OSError:
                continue

        prev = self._snapshot.get(slug, {})
        added = [p for p in current_files if p not in prev]
        changed = [p for p in current_files if p in prev and current_files[p] > prev[p] + 0.5]
        removed = [p for p in prev if p not in current_files]

        if not (added or changed or removed):
            return  # no diff

        # Diff rilevato: assicurati sezione esista in KB + reindex
        try:
            existing_sections = {s.slug: s for s in KnowledgeManager.list_sections()}
            if slug not in existing_sections:
                print(f"[kb-watcher] creo sezione '{slug}' da cartella {sec_dir.name}")
                KnowledgeManager.create_section(sec_dir.name)
            if added or changed:
                files_to_add = [Path(p) for p in (added + changed)]
                print(f"[kb-watcher] ingest {len(files_to_add)} file(s) → {slug}")
                try:
                    KnowledgeManager.add_files(slug, files_to_add)
                    # add_files include reindex (vedi kb/manager.py)
                except Exception as e:
                    print(f"[kb-watcher] add_files fallita {slug}: {e}")
            if removed:
                # add_files non gestisce remove. Forziamo reindex completo.
                print(f"[kb-watcher] {len(removed)} file rimossi da {slug} → reindex")
                try:
                    KnowledgeManager.reindex(slug, verbose=False)
                except Exception as e:
                    print(f"[kb-watcher] reindex fallito {slug}: {e}")
        except Exception as e:
            print(f"[kb-watcher] sync section {slug} error: {e}")

        # Aggiorna snapshot
        self._snapshot[slug] = current_files


def start_kb_watcher(root: Path | None = None) -> KBFolderWatcher:
    """Helper: avvia watcher con root personalizzato o default."""
    watcher = KBFolderWatcher(root or DEFAULT_DROP_FOLDER)
    watcher.start()
    return watcher
