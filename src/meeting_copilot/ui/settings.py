"""Settings panel — tweak runtime senza editare config.py.

Sezioni:
- Profili: low-power / balanced / max-quality (override modelli + soglie)
- LLM models: composer + analyzer (dropdown)
- RAG: embedding model, reranker on/off, hybrid retrieval on/off, HyDE on/off
- Watcher: soglie indicator/hook/auto (slider) — live preview
- Telemetry: on/off, view stats, purge, recompute, reset adaptive
- Help: re-run onboarding, open KB folder, view config file

I cambi vengono salvati in `~/.meeting_copilot/user_settings.json` come override
del config.py default. Letti al boot della app.
"""
from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .theme import THEME, qss


_USER_SETTINGS_FILE = Path.home() / ".meeting_copilot" / "user_settings.json"


# ─────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────
def load_user_settings() -> dict:
    if not _USER_SETTINGS_FILE.exists():
        return {}
    try:
        return json.loads(_USER_SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_user_settings(d: dict) -> None:
    _USER_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _USER_SETTINGS_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")


def apply_user_settings_to_config() -> None:
    """Override di config.py con i valori user. Chiamato all'avvio.
    NB: i dataclass sono frozen, usiamo object.__setattr__ per bypass."""
    settings = load_user_settings()
    if not settings:
        return

    from .. import config

    targets = {
        "stt.model_size": (config.STT, "model_size"),
        "stt.backend": (config.STT, "backend"),
        "llm.analyzer_model": (config.LLM, "analyzer_model"),
        "llm.composer_model": (config.LLM, "composer_model"),
        "rag.embedding_model": (config.RAG, "embedding_model"),
        "rag.rerank_enabled": (config.RAG, "rerank_enabled"),
        "rag.rerank_model": (config.RAG, "rerank_model"),
        "rag.hybrid_retrieval_enabled": (config.RAG, "hybrid_retrieval_enabled"),
        "rag.hyde_enabled": (config.RAG, "hyde_enabled"),
        "rag.parser_preference": (config.RAG, "parser_preference"),
        "rag.local_vector_backend": (config.RAG, "local_vector_backend"),
        "watcher.threshold_indicator": (config.WATCHER, "threshold_indicator"),
        "watcher.threshold_hook": (config.WATCHER, "threshold_hook"),
        "watcher.threshold_auto": (config.WATCHER, "threshold_auto"),
        "telemetry.enabled": (config.TELEMETRY, "enabled"),
    }
    for key, value in settings.items():
        if key in targets:
            obj, attr = targets[key]
            try:
                object.__setattr__(obj, attr, value)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
# Profiles
# ─────────────────────────────────────────────────────────────
PROFILES = {
    "low-power": {
        "stt.model_size": "turbo",
        "llm.analyzer_model": "llama3.2:3b",
        "llm.composer_model": "qwen2.5:7b-instruct",
        "rag.rerank_enabled": False,
        "rag.hyde_enabled": False,
        "rag.hybrid_retrieval_enabled": True,
    },
    "balanced": {
        "stt.model_size": "turbo",
        "llm.analyzer_model": "llama3.2:3b",
        "llm.composer_model": "qwen2.5:7b-instruct",
        "rag.rerank_enabled": True,
        "rag.hyde_enabled": False,
        "rag.hybrid_retrieval_enabled": True,
    },
    "max-quality": {
        "stt.model_size": "large-v3",
        "llm.analyzer_model": "llama3.2:3b",
        "llm.composer_model": "qwen2.5:14b-instruct",
        "rag.rerank_enabled": True,
        "rag.hyde_enabled": True,
        "rag.hybrid_retrieval_enabled": True,
        "rag.rerank_model": "BAAI/bge-reranker-v2-m3",
    },
}


# ─────────────────────────────────────────────────────────────
# Main dialog
# ─────────────────────────────────────────────────────────────
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Meeting Copilot — Impostazioni")
        self.resize(680, 580)
        self.setStyleSheet(
            f"QDialog {{ background-color: {THEME.palette.bg_elevated}; "
            f"color: {THEME.palette.text_primary}; }}"
            f"QTabWidget::pane {{ border: none; }}"
            f"QTabBar::tab {{ background: transparent; color: {THEME.palette.text_secondary}; "
            f"padding: 8px 16px; border: none; }}"
            f"QTabBar::tab:selected {{ color: {THEME.palette.text_primary}; "
            f"border-bottom: 2px solid {THEME.palette.accent_primary}; }}"
            f"QGroupBox {{ color: {THEME.palette.text_secondary}; "
            f"font-size: 11px; font-weight: 600; "
            f"border: 1px solid {THEME.palette.border_subtle}; "
            f"border-radius: 8px; margin-top: 8px; padding-top: 12px; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; "
            f"padding: 0 8px; }}"
            f"QLabel {{ color: {THEME.palette.text_primary}; }}"
            f"QComboBox {{ background: {THEME.palette.bg_surface}; "
            f"color: {THEME.palette.text_primary}; "
            f"padding: 6px 10px; border: 1px solid {THEME.palette.border_subtle}; "
            f"border-radius: 6px; min-width: 200px; }}"
            f"QSlider::groove:horizontal {{ height: 4px; background: {THEME.palette.bg_surface}; "
            f"border-radius: 2px; }}"
            f"QSlider::handle:horizontal {{ background: {THEME.palette.accent_primary}; "
            f"width: 14px; margin: -5px 0; border-radius: 7px; }}"
            f"QCheckBox {{ color: {THEME.palette.text_primary}; }}"
        )

        self._current = load_user_settings()

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(12)

        # Header
        title = QLabel("Impostazioni")
        title.setStyleSheet(qss.title_h1())
        root.addWidget(title)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_profiles(), "Profili")
        self.tabs.addTab(self._tab_llm(), "LLM")
        self.tabs.addTab(self._tab_rag(), "RAG")
        self.tabs.addTab(self._tab_watcher(), "Watcher")
        self.tabs.addTab(self._tab_privacy(), "Privacy & Help")
        root.addWidget(self.tabs, 1)

        # Footer
        footer = QHBoxLayout()
        footer.addStretch()
        btn_close = QPushButton("Chiudi")
        btn_close.setProperty("class", "secondary")
        btn_close.setStyleSheet(qss.secondary_button())
        btn_close.clicked.connect(self.accept)
        footer.addWidget(btn_close)

        btn_save = QPushButton("Salva e applica")
        btn_save.setObjectName("primary")
        btn_save.setStyleSheet(qss.primary_button())
        btn_save.clicked.connect(self._on_save)
        footer.addWidget(btn_save)

        root.addLayout(footer)

    # ── Tabs ──
    def _tab_profiles(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setSpacing(12)
        info = QLabel(
            "Scegli un profilo pre-configurato. Sovrascrive le impostazioni "
            "individuali nei tab successivi. Puoi sempre tweak manualmente dopo."
        )
        info.setWordWrap(True)
        info.setStyleSheet(qss.body_dim())
        l.addWidget(info)

        for key, descr in [
            ("low-power", "Massima velocità, RAM minima. STT turbo + qwen2.5:7b. No rerank."),
            ("balanced", "Bilanciato. STT turbo + qwen3:8b + reranker. Default."),
            ("max-quality", "Qualità top. STT large-v3 + qwen3:14b + reranker + HyDE. Più lento."),
        ]:
            box = QFrame()
            bl = QVBoxLayout(box)
            bl.setContentsMargins(12, 12, 12, 12)
            box.setStyleSheet(
                f"QFrame {{ background: {THEME.palette.bg_surface}; "
                f"border: 1px solid {THEME.palette.border_subtle}; border-radius: 8px; }}"
            )
            t = QLabel(key)
            t.setStyleSheet(qss.title_h2())
            bl.addWidget(t)
            d = QLabel(descr)
            d.setWordWrap(True)
            d.setStyleSheet(qss.body_dim())
            bl.addWidget(d)
            btn = QPushButton(f"Attiva profilo «{key}»")
            btn.setProperty("class", "secondary")
            btn.setStyleSheet(qss.secondary_button())
            btn.clicked.connect(lambda _, k=key: self._apply_profile(k))
            bl.addWidget(btn, alignment=Qt.AlignmentFlag.AlignLeft)
            l.addWidget(box)

        l.addStretch()
        return w

    def _tab_llm(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setSpacing(12)

        self.cb_analyzer = QComboBox()
        self.cb_analyzer.addItems(["llama3.2:3b", "phi3.5", "phi4-mini", "qwen2.5:3b"])
        cur = self._current.get("llm.analyzer_model", "llama3.2:3b")
        if self.cb_analyzer.findText(cur) >= 0:
            self.cb_analyzer.setCurrentText(cur)
        form.addRow("Analyzer model:", self.cb_analyzer)

        self.cb_composer = QComboBox()
        # NB: solo modelli VERIFICATI disponibili su Ollama registry
        self.cb_composer.addItems([
            "qwen2.5:7b-instruct",       # DEFAULT, 4.5GB
            "qwen2.5:14b-instruct",      # 9GB, qualità top
            "llama3.1:8b-instruct",      # 5GB, alternativa
            "mistral:7b-instruct",       # 4GB, alternativa
        ])
        cur = self._current.get("llm.composer_model", "qwen2.5:7b-instruct")
        if self.cb_composer.findText(cur) >= 0:
            self.cb_composer.setCurrentText(cur)
        form.addRow("Composer model:", self.cb_composer)

        note = QLabel("Cambiare modello richiede `ollama pull <modello>` prima.")
        note.setStyleSheet(qss.body_dim() + "font-size:11px;")
        form.addRow("", note)

        return w

    def _tab_rag(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setSpacing(12)

        self.cb_embed = QComboBox()
        self.cb_embed.addItems(["nomic-embed-text", "bge-m3", "mxbai-embed-large"])
        cur = self._current.get("rag.embedding_model", "bge-m3")
        if self.cb_embed.findText(cur) >= 0:
            self.cb_embed.setCurrentText(cur)
        form.addRow("Embedding model:", self.cb_embed)

        self.cb_rerank = QComboBox()
        self.cb_rerank.addItems([
            "mixedbread-ai/mxbai-rerank-xsmall-v1",
            "BAAI/bge-reranker-v2-m3",
            "BAAI/bge-reranker-base",
        ])
        cur = self._current.get("rag.rerank_model", "BAAI/bge-reranker-v2-m3")
        if self.cb_rerank.findText(cur) >= 0:
            self.cb_rerank.setCurrentText(cur)
        form.addRow("Reranker model:", self.cb_rerank)

        self.chk_rerank = QCheckBox("Attiva reranker cross-encoder")
        self.chk_rerank.setChecked(self._current.get("rag.rerank_enabled", True))
        form.addRow("", self.chk_rerank)

        self.chk_hybrid = QCheckBox("Hybrid retrieval (BM25 + dense con RRF)")
        self.chk_hybrid.setChecked(self._current.get("rag.hybrid_retrieval_enabled", True))
        form.addRow("", self.chk_hybrid)

        self.chk_hyde = QCheckBox("HyDE query expansion (più lento, +recall)")
        self.chk_hyde.setChecked(self._current.get("rag.hyde_enabled", False))
        form.addRow("", self.chk_hyde)

        self.cb_parser = QComboBox()
        self.cb_parser.addItems(["auto", "markitdown", "specific"])
        cur = self._current.get("rag.parser_preference", "auto")
        if self.cb_parser.findText(cur) >= 0:
            self.cb_parser.setCurrentText(cur)
        form.addRow("Parser preference:", self.cb_parser)

        self.cb_vec = QComboBox()
        self.cb_vec.addItems(["chromadb", "lancedb"])
        cur = self._current.get("rag.local_vector_backend", "chromadb")
        if self.cb_vec.findText(cur) >= 0:
            self.cb_vec.setCurrentText(cur)
        form.addRow("Vector backend:", self.cb_vec)

        return w

    def _tab_watcher(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setSpacing(12)

        info = QLabel(
            "Soglie del Watcher Agent. Più alte = sistema più conservativo "
            "(meno suggerimenti). Più basse = più proattivo (rischio rumore)."
        )
        info.setWordWrap(True)
        info.setStyleSheet(qss.body_dim())
        l.addWidget(info)

        from ..config import WATCHER

        self.s_indicator, self.lbl_ind_val = self._slider_row(
            l, "Threshold indicator",
            self._current.get("watcher.threshold_indicator", WATCHER.threshold_indicator),
        )
        self.s_hook, self.lbl_hook_val = self._slider_row(
            l, "Threshold hook (gancio cliccabile)",
            self._current.get("watcher.threshold_hook", WATCHER.threshold_hook),
        )
        self.s_auto, self.lbl_auto_val = self._slider_row(
            l, "Threshold auto (auto-fire)",
            self._current.get("watcher.threshold_auto", WATCHER.threshold_auto),
        )

        adaptive_note = QLabel(
            "Le soglie possono essere aggiornate automaticamente dal sistema "
            "(telemetry/learner). Vedi tab Privacy → Telemetry."
        )
        adaptive_note.setWordWrap(True)
        adaptive_note.setStyleSheet(qss.body_dim() + "font-size:11px;")
        l.addWidget(adaptive_note)

        l.addStretch()
        return w

    def _slider_row(self, parent_layout, label: str, value: float) -> tuple:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(240)
        row.addWidget(lbl)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(0, 100)
        s.setValue(int(value * 100))
        row.addWidget(s, 1)
        v = QLabel(f"{value:.2f}")
        v.setMinimumWidth(48)
        v.setStyleSheet(f"color:{THEME.palette.accent_primary}; font-weight:600;")
        s.valueChanged.connect(lambda i, lab=v: lab.setText(f"{i/100:.2f}"))
        row.addWidget(v)
        parent_layout.addLayout(row)
        return (s, v)

    def _tab_privacy(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setSpacing(12)

        # Telemetry
        gb = QGroupBox("Telemetry & adaptive learning")
        gbl = QVBoxLayout(gb)
        self.chk_telemetry = QCheckBox("Telemetry attiva (locale, on-device, opt-in)")
        self.chk_telemetry.setChecked(self._current.get("telemetry.enabled", True))
        gbl.addWidget(self.chk_telemetry)
        info = QLabel(
            "Salva eventi del Watcher (decisioni, click sui ganci, thumbs) in un "
            "database SQLite locale (~/.meeting_copilot/telemetry.db) per migliorare "
            "le soglie nel tempo. Mai inviato a nessun server."
        )
        info.setWordWrap(True)
        info.setStyleSheet(qss.body_dim() + "font-size:12px;")
        gbl.addWidget(info)
        row = QHBoxLayout()
        for label, action in [
            ("Vedi statistiche", self._show_telemetry_stats),
            ("Cancella tutto", self._purge_telemetry),
            ("Reset soglie adaptive", self._reset_adaptive),
        ]:
            b = QPushButton(label)
            b.setProperty("class", "secondary")
            b.setStyleSheet(qss.secondary_button())
            b.clicked.connect(action)
            row.addWidget(b)
        row.addStretch()
        gbl.addLayout(row)
        l.addWidget(gb)

        # Help
        gb2 = QGroupBox("Help")
        gbl2 = QVBoxLayout(gb2)
        for label, action in [
            ("Rifai onboarding", self._rerun_onboarding),
            ("Apri cartella KB", self._open_kb_folder),
            ("Apri SETUP.md", self._open_setup),
        ]:
            b = QPushButton(label)
            b.setProperty("class", "secondary")
            b.setStyleSheet(qss.secondary_button())
            b.clicked.connect(action)
            gbl2.addWidget(b, alignment=Qt.AlignmentFlag.AlignLeft)
        l.addWidget(gb2)

        l.addStretch()
        return w

    # ── Actions ──
    def _apply_profile(self, name: str) -> None:
        profile = PROFILES.get(name)
        if not profile:
            return
        self._current.update(profile)
        save_user_settings(self._current)
        QMessageBox.information(
            self, "Profilo applicato",
            f"Profilo «{name}» salvato. Riapri il tool per applicare i cambi modello.",
        )

    def _on_save(self) -> None:
        # Raccoglie da tutti i tab
        self._current["llm.analyzer_model"] = self.cb_analyzer.currentText()
        self._current["llm.composer_model"] = self.cb_composer.currentText()
        self._current["rag.embedding_model"] = self.cb_embed.currentText()
        self._current["rag.rerank_model"] = self.cb_rerank.currentText()
        self._current["rag.rerank_enabled"] = self.chk_rerank.isChecked()
        self._current["rag.hybrid_retrieval_enabled"] = self.chk_hybrid.isChecked()
        self._current["rag.hyde_enabled"] = self.chk_hyde.isChecked()
        self._current["rag.parser_preference"] = self.cb_parser.currentText()
        self._current["rag.local_vector_backend"] = self.cb_vec.currentText()
        self._current["watcher.threshold_indicator"] = self.s_indicator.value() / 100
        self._current["watcher.threshold_hook"] = self.s_hook.value() / 100
        self._current["watcher.threshold_auto"] = self.s_auto.value() / 100
        self._current["telemetry.enabled"] = self.chk_telemetry.isChecked()

        save_user_settings(self._current)
        apply_user_settings_to_config()
        QMessageBox.information(
            self, "Impostazioni salvate",
            "Salvate. Alcuni cambi (modelli LLM, embedding) richiedono riavvio "
            "dell'app per essere applicati completamente.",
        )

    def _show_telemetry_stats(self) -> None:
        try:
            from ..telemetry.store import TelemetryStore
            store = TelemetryStore.instance()
            d = store.query("SELECT COUNT(*) AS n FROM decisions")[0]["n"]
            c = store.query("SELECT COUNT(*) AS n FROM user_feedback WHERE action='clicked'")[0]["n"]
            s = store.query("SELECT COUNT(*) AS n FROM user_feedback WHERE action='shown'")[0]["n"]
            up = store.query("SELECT COUNT(*) AS n FROM user_feedback WHERE action='thumbs_up'")[0]["n"]
            dn = store.query("SELECT COUNT(*) AS n FROM user_feedback WHERE action='thumbs_down'")[0]["n"]
            rate = c / s * 100 if s > 0 else 0
            QMessageBox.information(
                self, "Stats telemetria",
                f"Decisioni totali: {d}\n"
                f"Ganci mostrati: {s}\n"
                f"Click rate: {rate:.1f}%\n"
                f"Thumbs up: {up}\n"
                f"Thumbs down: {dn}",
            )
        except Exception as e:
            QMessageBox.warning(self, "Errore", f"Impossibile leggere telemetry: {e}")

    def _purge_telemetry(self) -> None:
        if QMessageBox.question(
            self, "Cancella telemetria?",
            "Tutti i dati telemetrici locali verranno cancellati. Continua?",
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            from ..telemetry.store import DEFAULT_DB_PATH
            from ..telemetry.learner import STATE_FILE
            if DEFAULT_DB_PATH.exists():
                DEFAULT_DB_PATH.unlink()
            if STATE_FILE.exists():
                STATE_FILE.unlink()
            QMessageBox.information(self, "Fatto", "Telemetria cancellata.")
        except Exception as e:
            QMessageBox.warning(self, "Errore", str(e))

    def _reset_adaptive(self) -> None:
        try:
            from ..telemetry.learner import reset_adaptive_state
            state = reset_adaptive_state()
            QMessageBox.information(
                self, "Reset", f"Soglie reset ai default statici.\n"
                f"hook={state.threshold_hook:.2f}, auto={state.threshold_auto:.2f}"
            )
        except Exception as e:
            QMessageBox.warning(self, "Errore", str(e))

    def _rerun_onboarding(self) -> None:
        from .onboarding import reset_onboarded, run_onboarding_if_needed
        reset_onboarded()
        run_onboarding_if_needed(force=True)

    def _open_kb_folder(self) -> None:
        import subprocess
        from ..kb import paths as kb_paths
        subprocess.run(["open", str(kb_paths.KB_ROOT)])

    def _open_setup(self) -> None:
        import subprocess
        try:
            setup_md = Path(__file__).resolve().parents[3] / "SETUP.md"
            if setup_md.exists():
                subprocess.run(["open", str(setup_md)])
        except Exception:
            pass


def open_settings(parent=None) -> None:
    dlg = SettingsDialog(parent=parent)
    dlg.exec()
