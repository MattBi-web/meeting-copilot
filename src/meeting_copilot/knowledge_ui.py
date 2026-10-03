"""Knowledge Manager — finestra PyQt6 dedicata.

Layout:
    ┌─────────────────────────────────────────────────────────┐
    │  Meeting Copilot — Knowledge Manager                  ✕ │
    ├──────────────┬──────────────────────────────────────────┤
    │  Sezioni     │  ▸ pitch_productb                       │
    │  ───────     │     ─────────────────                     │
    │  ● pitch_x   │     5 file · 47 chunks                    │
    │  ○ cliente_y │     Ultima reindex: 14/05 16:32           │
    │  ○ obiezioni │                                            │
    │  ○ tecnico   │     ┌─ Sorgenti ────────────────┐         │
    │              │     │ slide.pdf      [12 chunks]│         │
    │  [+ Nuova]   │     │ pricing.xlsx   [8 chunks] │         │
    │              │     │ pitch.md       [15 chunks]│         │
    │              │     │ _urls.txt      [12 chunks]│         │
    │              │     └────────────────────────────┘        │
    │              │                                            │
    │              │     [📁 Apri cartella]  [⟳ Re-index]      │
    │              │     [✕ Elimina sezione]                   │
    │              │                                            │
    │              │     ─── Test search ───                   │
    │              │     [____________________] [Cerca]        │
    │              │     (risultati qui)                       │
    └──────────────┴──────────────────────────────────────────┘
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .kb.manager import KnowledgeManager
from .kb import paths


# ─────────────────────────────────────────────────────────────
# Worker thread per reindex (non blocca UI)
# ─────────────────────────────────────────────────────────────
class _ReindexWorker(QThread):
    done = pyqtSignal(str, dict, str)   # slug, meta, error_msg ("" se ok)

    def __init__(self, slug: str):
        super().__init__()
        self.slug = slug

    def run(self) -> None:
        try:
            meta = KnowledgeManager.reindex(self.slug, verbose=False)
            self.done.emit(self.slug, meta, "")
        except Exception as e:
            self.done.emit(self.slug, {}, str(e))


# ─────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────
class KnowledgeManagerWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meeting Copilot — Knowledge Manager")
        self.resize(900, 600)
        self.setAcceptDrops(True)

        self._current_slug: str | None = None
        self._worker: _ReindexWorker | None = None

        self.setStyleSheet("""
            QWidget { background-color: #0F1116; color: #E5E7EB; font-size: 13px; }
            QListWidget { background-color: #15171C; border: 1px solid #2A2D35; border-radius: 6px; padding: 4px; }
            QListWidget::item { padding: 6px 8px; border-radius: 4px; }
            QListWidget::item:selected { background-color: #1F2937; color: #fff; }
            QListWidget::item:hover { background-color: #1A1D24; }
            QPushButton {
                background-color: #1F2937; color: #E5E7EB; border: 1px solid #374151;
                padding: 7px 14px; border-radius: 6px; font-weight: 500;
            }
            QPushButton:hover { background-color: #2D3748; }
            QPushButton:disabled { color: #6B7280; }
            QPushButton#primary { background-color: #2563EB; border-color: #3B82F6; }
            QPushButton#primary:hover { background-color: #3B82F6; }
            QPushButton#danger { background-color: #7F1D1D; border-color: #991B1B; }
            QPushButton#danger:hover { background-color: #991B1B; }
            QLineEdit, QTextEdit {
                background-color: #15171C; color: #E5E7EB; border: 1px solid #2A2D35;
                border-radius: 6px; padding: 6px 8px;
            }
            QLabel.h2 { font-size: 16px; font-weight: 600; color: #fff; }
            QLabel.meta { color: #9CA3AF; font-size: 12px; }
            QLabel.section { color: #6B7280; font-size: 10px; font-weight: 600;
                             text-transform: uppercase; letter-spacing: 1px; }
            QCheckBox { color: #E5E7EB; }
        """)

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        root.addWidget(splitter)

        # ── Sidebar (sx) ──
        side = QFrame()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)

        lbl_side = QLabel("SEZIONI")
        lbl_side.setProperty("class", "section")
        lbl_side.setStyleSheet("color: #6B7280; font-size: 10px; font-weight: 600; letter-spacing: 1px; margin-bottom: 4px;")
        sl.addWidget(lbl_side)

        self.list_sections = QListWidget()
        self.list_sections.itemSelectionChanged.connect(self._on_section_selected)
        sl.addWidget(self.list_sections, 1)

        self.btn_new = QPushButton("+ Nuova sezione")
        self.btn_new.setObjectName("primary")
        self.btn_new.clicked.connect(self._on_new_section)
        sl.addWidget(self.btn_new)

        side.setMinimumWidth(220)
        splitter.addWidget(side)

        # ── Detail panel (dx) ──
        self.detail = self._build_detail_panel()
        splitter.addWidget(self.detail)

        splitter.setSizes([240, 660])

        # Iniziale
        self._refresh_sections_list()

    # ─────────────────────────────────────────────────────────
    # Detail panel
    # ─────────────────────────────────────────────────────────
    def _build_detail_panel(self) -> QWidget:
        w = QFrame()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(8)

        self.lbl_title = QLabel("Seleziona una sezione")
        self.lbl_title.setProperty("class", "h2")
        self.lbl_title.setStyleSheet("font-size: 18px; font-weight: 600; color: #fff;")
        l.addWidget(self.lbl_title)

        self.lbl_meta = QLabel("")
        self.lbl_meta.setStyleSheet("color: #9CA3AF; font-size: 12px;")
        l.addWidget(self.lbl_meta)

        self.chk_active = QCheckBox("Sezione attiva per la call corrente")
        self.chk_active.stateChanged.connect(self._on_toggle_active)
        l.addWidget(self.chk_active)

        # Sorgenti
        lbl_src = QLabel("SORGENTI")
        lbl_src.setStyleSheet("color: #6B7280; font-size: 10px; font-weight: 600; letter-spacing: 1px; margin-top: 12px;")
        l.addWidget(lbl_src)

        self.list_sources = QListWidget()
        self.list_sources.setMinimumHeight(150)
        l.addWidget(self.list_sources)

        # Buttons row
        btns = QHBoxLayout()
        self.btn_add_files = QPushButton("➕ Aggiungi file…")
        self.btn_add_files.clicked.connect(self._on_add_files)
        btns.addWidget(self.btn_add_files)

        self.btn_open_folder = QPushButton("📁 Apri cartella")
        self.btn_open_folder.clicked.connect(self._on_open_folder)
        btns.addWidget(self.btn_open_folder)

        self.btn_reindex = QPushButton("⟳ Re-index")
        self.btn_reindex.setObjectName("primary")
        self.btn_reindex.clicked.connect(self._on_reindex)
        btns.addWidget(self.btn_reindex)

        btns.addStretch()

        self.btn_delete = QPushButton("✕ Elimina")
        self.btn_delete.setObjectName("danger")
        self.btn_delete.clicked.connect(self._on_delete)
        btns.addWidget(self.btn_delete)

        l.addLayout(btns)

        # Test search
        lbl_search = QLabel("TEST SEARCH")
        lbl_search.setStyleSheet("color: #6B7280; font-size: 10px; font-weight: 600; letter-spacing: 1px; margin-top: 12px;")
        l.addWidget(lbl_search)

        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Query naturale (es. 'qual è la latenza dei voicebot?')")
        self.search_input.returnPressed.connect(self._on_search)
        search_row.addWidget(self.search_input, 1)
        self.btn_search = QPushButton("Cerca")
        self.btn_search.clicked.connect(self._on_search)
        search_row.addWidget(self.btn_search)
        l.addLayout(search_row)

        self.search_results = QTextEdit()
        self.search_results.setReadOnly(True)
        self.search_results.setMinimumHeight(140)
        l.addWidget(self.search_results, 1)

        self._set_detail_enabled(False)
        return w

    def _set_detail_enabled(self, enabled: bool) -> None:
        for w in (self.chk_active, self.list_sources, self.btn_add_files,
                  self.btn_open_folder, self.btn_reindex, self.btn_delete,
                  self.search_input, self.btn_search):
            w.setEnabled(enabled)

    # ─────────────────────────────────────────────────────────
    # Refresh / data binding
    # ─────────────────────────────────────────────────────────
    def _refresh_sections_list(self) -> None:
        self.list_sections.clear()
        active = set(KnowledgeManager.get_active_sections())
        sections = KnowledgeManager.list_sections()
        for s in sections:
            text = f"{'●' if s.slug in active else '○'} {s.slug}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, s.slug)
            self.list_sections.addItem(item)

    def _refresh_detail(self) -> None:
        if not self._current_slug:
            self._set_detail_enabled(False)
            self.lbl_title.setText("Seleziona una sezione")
            self.lbl_meta.setText("")
            self.list_sources.clear()
            return

        slug = self._current_slug
        sections = {s.slug: s for s in KnowledgeManager.list_sections()}
        s = sections.get(slug)
        if s is None:
            return

        self.lbl_title.setText(slug)
        self.lbl_meta.setText(
            f"{s.n_sources} sorgenti · {s.n_chunks} chunks · "
            f"Ultima reindex: {s.last_reindex or '(mai)'}"
        )
        active = KnowledgeManager.get_active_sections()
        self.chk_active.blockSignals(True)
        self.chk_active.setChecked(slug in active)
        self.chk_active.blockSignals(False)

        self._refresh_sources_list(slug)
        self._set_detail_enabled(True)

    def _refresh_sources_list(self, slug: str) -> None:
        self.list_sources.clear()
        src_dir = paths.sources_dir(slug)
        if not src_dir.exists():
            return
        for entry in sorted(src_dir.iterdir()):
            if entry.is_dir() or entry.name.startswith("."):
                continue
            size_kb = entry.stat().st_size / 1024
            label = f"{entry.name}  [{size_kb:.1f} KB]"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, str(entry))
            self.list_sources.addItem(item)

    # ─────────────────────────────────────────────────────────
    # Slots
    # ─────────────────────────────────────────────────────────
    def _on_section_selected(self) -> None:
        items = self.list_sections.selectedItems()
        if not items:
            self._current_slug = None
        else:
            self._current_slug = items[0].data(Qt.ItemDataRole.UserRole)
        self._refresh_detail()

    def _on_new_section(self) -> None:
        name, ok = QInputDialog.getText(
            self, "Nuova sezione", "Nome (es. 'Pitch ProductB'):"
        )
        if not ok or not name.strip():
            return
        section = KnowledgeManager.create_section(name.strip())
        self._refresh_sections_list()
        # Auto-seleziona quella nuova
        for i in range(self.list_sections.count()):
            if self.list_sections.item(i).data(Qt.ItemDataRole.UserRole) == section.slug:
                self.list_sections.setCurrentRow(i)
                break

    def _on_toggle_active(self, state) -> None:
        if not self._current_slug:
            return
        active = KnowledgeManager.get_active_sections()
        if state == Qt.CheckState.Checked.value:
            if self._current_slug not in active:
                active.append(self._current_slug)
        else:
            if self._current_slug in active:
                active.remove(self._current_slug)
        KnowledgeManager.set_active_sections(active)
        self._refresh_sections_list()
        # Mantieni selezione
        for i in range(self.list_sections.count()):
            if self.list_sections.item(i).data(Qt.ItemDataRole.UserRole) == self._current_slug:
                self.list_sections.setCurrentRow(i)
                break

    def _on_add_files(self) -> None:
        if not self._current_slug:
            return
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Scegli file da aggiungere",
            str(Path.home()),
            "Documenti (*.pdf *.docx *.pptx *.xlsx *.md *.txt);;Tutti (*)",
        )
        if not files:
            return
        copied = KnowledgeManager.add_files(self._current_slug, [Path(f) for f in files])
        self._refresh_sources_list(self._current_slug)
        QMessageBox.information(
            self, "File aggiunti",
            f"{len(copied)} file copiati in sources/.\nLancia ⟳ Re-index per indicizzarli."
        )

    def _on_open_folder(self) -> None:
        if not self._current_slug:
            return
        src_dir = paths.sources_dir(self._current_slug)
        src_dir.mkdir(parents=True, exist_ok=True)
        # macOS: open con Finder
        if sys.platform == "darwin":
            subprocess.run(["open", str(src_dir)])
        elif sys.platform == "win32":
            subprocess.run(["explorer", str(src_dir)])
        else:
            subprocess.run(["xdg-open", str(src_dir)])

    def _on_reindex(self) -> None:
        if not self._current_slug:
            return
        if self._worker and self._worker.isRunning():
            return
        self.btn_reindex.setEnabled(False)
        self.btn_reindex.setText("⟳ Reindexing…")
        self._worker = _ReindexWorker(self._current_slug)
        self._worker.done.connect(self._on_reindex_done)
        self._worker.start()

    @pyqtSlot(str, dict, str)
    def _on_reindex_done(self, slug: str, meta: dict, error: str) -> None:
        self.btn_reindex.setEnabled(True)
        self.btn_reindex.setText("⟳ Re-index")
        if error:
            QMessageBox.critical(self, "Reindex fallito", f"Errore:\n{error}")
        else:
            self._refresh_sections_list()
            # Mantieni selezione
            for i in range(self.list_sections.count()):
                if self.list_sections.item(i).data(Qt.ItemDataRole.UserRole) == slug:
                    self.list_sections.setCurrentRow(i)
                    break
            self._refresh_detail()

    def _on_delete(self) -> None:
        if not self._current_slug:
            return
        resp = QMessageBox.question(
            self, "Eliminare sezione?",
            f"Eliminare DEFINITIVAMENTE la sezione '{self._current_slug}'?\n"
            "Verranno cancellati sources + index. Operazione non reversibile.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        KnowledgeManager.delete_section(self._current_slug)
        self._current_slug = None
        self._refresh_sections_list()
        self._refresh_detail()

    def _on_search(self) -> None:
        if not self._current_slug:
            return
        q = self.search_input.text().strip()
        if not q:
            return
        try:
            hits = KnowledgeManager.search(q, sections=[self._current_slug], top_k=5)
        except Exception as e:
            self.search_results.setPlainText(f"Errore: {e}")
            return

        if not hits:
            self.search_results.setHtml("<p style='color:#9CA3AF;'>Nessun risultato.</p>")
            return

        html = ""
        for i, h in enumerate(hits, 1):
            html += (
                f"<div style='margin-bottom:14px;'>"
                f"<div style='color:#9CA3AF; font-size:11px; margin-bottom:4px;'>"
                f"#{i} · score={h.score:.3f} · "
                f"<span style='color:#F59E0B;'>{h.source}</span>"
                f"</div>"
                f"<div style='color:#E5E7EB;'>{_escape(h.text[:500])}"
                f"{'…' if len(h.text) > 500 else ''}</div>"
                f"</div>"
            )
        self.search_results.setHtml(html)

    # ─────────────────────────────────────────────────────────
    # Drag&Drop
    # ─────────────────────────────────────────────────────────
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and self._current_slug:
            e.acceptProposedAction()

    def dropEvent(self, e):
        if not self._current_slug:
            return
        files = [Path(url.toLocalFile()) for url in e.mimeData().urls() if url.isLocalFile()]
        if not files:
            return
        copied = KnowledgeManager.add_files(self._current_slug, files)
        self._refresh_sources_list(self._current_slug)
        QMessageBox.information(
            self, "Drag&Drop",
            f"{len(copied)} file aggiunti. Lancia ⟳ Re-index per indicizzarli."
        )


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace("\n", "<br>"))


# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────
def run_kb_ui() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    win = KnowledgeManagerWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_kb_ui()
