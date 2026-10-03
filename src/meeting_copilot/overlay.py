"""Overlay PyQt6 — finestra invisibile a screen share.

Layout:
    ┌────────────────────────────────┐
    │ ●●●                ↻ ✕         │  ← header (drag + chiudi)
    ├────────────────────────────────┤
    │                                │
    │  Trascrizione live              │
    │  (autoscroll)                   │
    │                                │
    ├────────────────────────────────┤
    │  Suggerimenti LLM (Fase 4)      │
    │  [placeholder per ora]          │
    └────────────────────────────────┘

Caratteristiche:
- Frameless (no barra titolo) + WindowStaysOnTopHint
- Background semi-trasparente, look "scuro pulito"
- Drag-by-mouse sull'header
- Hotkey: Cmd+Shift+T per mostra/nascondi, Esc per chiudi
- NSWindowSharingNone applicato dopo show()
"""
from __future__ import annotations

from datetime import datetime

from collections import deque

from PyQt6.QtCore import (
    QObject,
    QPoint,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from .macos_window import (
    make_invisible_to_screen_share,
    set_window_always_on_top_floating,
)
from .kb.manager import KnowledgeManager
from .llm.orchestrator import Orchestrator
from .llm.watcher import WatcherAgent, WatcherDecision
from .session import Session
from .global_hotkeys import GlobalHotkeys
from .ui.theme import THEME, qss


# ─────────────────────────────────────────────────────────────
# Bridge thread-safe per i segnali tra worker thread e UI Qt
# ─────────────────────────────────────────────────────────────
class TranscriptionBridge(QObject):
    """Bridge thread-safe. I worker thread (audio/STT) chiamano emit(),
    Qt instrada in modo sicuro l'aggiornamento sul thread della UI."""

    new_line = pyqtSignal(str, str, float, str)   # text, language, duration_s, speaker
    status = pyqtSignal(str)                  # messaggi di stato (errori, info)
    # Orchestrator → UI (cross-thread, sempre via Qt signal)
    llm_status = pyqtSignal(str, str)          # status_text, color_hex
    llm_token = pyqtSignal(str)                # streaming delta
    llm_complete = pyqtSignal(str, list)       # full_text, hits
    # Watcher → UI (Fase 6)
    watcher_decision = pyqtSignal(object)      # WatcherDecision
    watcher_token = pyqtSignal(str, str)       # decision_id, piece
    watcher_complete = pyqtSignal(str, str, list)  # decision_id, full_text, hits


# ─────────────────────────────────────────────────────────────
# Worker per retrieval RAG asincrono (non blocca UI)
# ─────────────────────────────────────────────────────────────
class _RAGWorker(QThread):
    """Esegue una query RAG su un thread separato. Riemette i risultati via signal."""

    done = pyqtSignal(list, str)   # list[SearchHit], error_msg

    def __init__(self, query: str, sections: list[str], top_k: int = 3):
        super().__init__()
        self.query = query
        self.sections = sections
        self.top_k = top_k

    def run(self) -> None:
        try:
            hits = KnowledgeManager.search(self.query, sections=self.sections, top_k=self.top_k)
            self.done.emit(hits, "")
        except Exception as e:
            self.done.emit([], str(e))


# ─────────────────────────────────────────────────────────────
# Header con bottoni custom (no decorazione di sistema)
# ─────────────────────────────────────────────────────────────
class _Header(QFrame):
    """Header sottile, drag-handle + bottoni mute/close."""

    def __init__(self, parent: "OverlayWindow"):
        super().__init__(parent)
        self._parent = parent
        self._drag_pos: QPoint | None = None

        self.setFixedHeight(40)
        # Header glass-style con tokens del theme
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {THEME.palette.bg_surface};
                border-top-left-radius: {THEME.radius.lg}px;
                border-top-right-radius: {THEME.radius.lg}px;
                border-bottom: 1px solid {THEME.palette.border_subtle};
            }}
            QLabel {{
                color: {THEME.palette.text_secondary};
                font-weight: 500;
                font-size: {THEME.typography.caption}px;
            }}
            {qss.header_button()}
        """)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 6, 0)
        lay.setSpacing(8)

        # Indicatore "stato"
        self._title = QLabel("● Meeting Copilot")
        self._title.setStyleSheet("color: #10B981; font-weight: 600; font-size: 11px;")
        lay.addWidget(self._title)

        lay.addStretch()

        # Bottone "💡 Ask" — triggera manualmente analyzer+composer adesso
        self.btn_ask = QPushButton("💡 Ask")
        self.btn_ask.setToolTip("Triggera Analyzer→Composer ora (anche Cmd+Shift+R)")
        self.btn_ask.clicked.connect(self._parent.trigger_rag_search)
        lay.addWidget(self.btn_ask)

        # Toggle Auto/Manual per orchestrator LLM
        self.btn_auto = QPushButton("◇ Auto")
        self.btn_auto.setCheckable(True)
        self.btn_auto.setToolTip("Auto: triggera ogni 8s | Manual: solo bottone/hotkey")
        self.btn_auto.toggled.connect(self._parent.set_auto_mode)
        lay.addWidget(self.btn_auto)

        # Bottone KB (selettore sezioni attive)
        self.btn_kb = QPushButton("📚 KB")
        self.btn_kb.setToolTip("Sezioni di knowledge attive — click per selezionare")
        self.btn_kb.clicked.connect(self._parent.show_kb_menu)
        lay.addWidget(self.btn_kb)

        # Bottone copia trascrizione
        self.btn_copy = QPushButton("⧉")
        self.btn_copy.setToolTip("Copia trascrizione corrente")
        self.btn_copy.clicked.connect(self._parent.copy_transcript)
        lay.addWidget(self.btn_copy)

        # Bottone mute (placeholder Fase 4)
        self.btn_mute = QPushButton("⏸")
        self.btn_mute.setToolTip("Pausa trascrizione (TODO)")
        self.btn_mute.setEnabled(False)
        lay.addWidget(self.btn_mute)

        # Bottone Settings (Fase 11)
        self.btn_settings = QPushButton("⚙")
        self.btn_settings.setToolTip("Impostazioni & Help")
        self.btn_settings.clicked.connect(self._parent.open_settings)
        lay.addWidget(self.btn_settings)

        # Bottone close
        self.btn_close = QPushButton("✕")
        self.btn_close.setToolTip("Chiudi (Esc)")
        self.btn_close.clicked.connect(self._parent.close)
        lay.addWidget(self.btn_close)

    # --- drag ---
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = ev.globalPosition().toPoint() - self._parent.frameGeometry().topLeft()
            ev.accept()

    def mouseMoveEvent(self, ev):
        if self._drag_pos is not None and ev.buttons() & Qt.MouseButton.LeftButton:
            self._parent.move(ev.globalPosition().toPoint() - self._drag_pos)
            ev.accept()

    def mouseReleaseEvent(self, ev):
        self._drag_pos = None
        ev.accept()

    def set_status(self, label: str, color: str = "#10B981") -> None:
        self._title.setText(label)
        self._title.setStyleSheet(f"color: {color}; font-weight: 600; font-size: 11px;")


# ─────────────────────────────────────────────────────────────
# Main overlay window
# ─────────────────────────────────────────────────────────────
class OverlayWindow(QWidget):
    """Finestra overlay principale."""

    def __init__(self, bridge: TranscriptionBridge):
        super().__init__()
        self.bridge = bridge

        # Buffer delle ultime N righe trascritte (per retrieval RAG / LLM).
        # Tieni le ultime ~30 righe (≈ ultimi 1-2 minuti di parlato).
        self._recent_lines: deque[str] = deque(maxlen=30)
        self._rag_worker: _RAGWorker | None = None

        # Orchestrator LLM (Fase 4): inizializzato pigro la prima volta che serve
        self._orchestrator: Orchestrator | None = None
        self._composer_buffer: list[str] = []

        # Session (Fase 5): traccia tutto quello che succede in questa call
        self._session: Session = Session()
        self._session.set_initial_sections(KnowledgeManager.get_active_sections())
        # Buffer per la sessione: l'ultima query+chunks attiva per future suggestion logging
        self._last_query: str = ""
        self._last_category: str = ""
        self._last_confidence: float = 0.0

        # Watcher Agent (Fase 6) — auto-trigger intelligente
        self._watcher: WatcherAgent | None = None
        # Stato hooks attivi (decision_id → label per pannello hooks)
        self._active_hooks: dict[str, str] = {}
        # ID della decisione attualmente visualizzata nel pannello suggerimenti
        self._current_displayed_decision: str | None = None

        # Throttle del rendering durante streaming: setHtml è pesante
        # (ridisegna intero pannello). Aggiorniamo max 10 volte al secondo.
        self._stream_dirty = False
        self._stream_timer = QTimer(self)
        self._stream_timer.setInterval(100)  # ms
        self._stream_timer.timeout.connect(self._flush_stream)

        # Window flags: frameless + always-on-top.
        # NOTA: Qt.WindowType.Tool su macOS può rendere la finestra invisibile
        # quando l'app è lanciata da terminale senza focus → rimosso.
        # L'app apparirà nel Dock; per nasconderla useremo
        # NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory).
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # Overlay XL per leggere comodamente 2 suggerimenti side-by-side + chat
        self.setMinimumSize(640, 560)
        self.resize(860, 980)

        # ── Container con sfondo arrotondato (theme-driven) ──
        container = QFrame(self)
        container.setObjectName("container")
        container.setStyleSheet(
            qss.container_root() + qss.text_view()
        )

        # Layout principale dentro il container
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(container)

        inner = QVBoxLayout(container)
        inner.setContentsMargins(0, 0, 0, 8)
        inner.setSpacing(0)

        # Header
        self.header = _Header(self)
        inner.addWidget(self.header)

        # Splitter trascrizione / suggerimenti
        splitter = QSplitter(Qt.Orientation.Vertical, container)
        splitter.setHandleWidth(2)
        splitter.setChildrenCollapsible(False)

        # Pannello trascrizione
        trans_wrap = QFrame(container)
        tlay = QVBoxLayout(trans_wrap)
        tlay.setContentsMargins(0, 6, 0, 0)
        tlay.setSpacing(0)
        lbl_t = QLabel("Trascrizione live")
        lbl_t.setProperty("class", "section")
        lbl_t.setStyleSheet("color: #6B7280; font-size: 10px; font-weight: 600; "
                            "text-transform: uppercase; letter-spacing: 1px; padding: 6px 12px 2px 12px;")
        tlay.addWidget(lbl_t)
        self.transcript_view = QTextEdit(trans_wrap)
        self.transcript_view.setReadOnly(True)
        # Usa il font di sistema di macOS (".AppleSystemUIFont" o equivalente).
        # Evita il warning "missing font family SF Pro Text".
        sys_font = QFont()
        sys_font.setPointSize(13)
        self.transcript_view.setFont(sys_font)
        tlay.addWidget(self.transcript_view, 1)
        splitter.addWidget(trans_wrap)

        # Pannello GANCI (Fase 6): suggerimenti cliccabili quando score 0.55-0.75
        hooks_wrap = QFrame(container)
        hlay = QVBoxLayout(hooks_wrap)
        hlay.setContentsMargins(0, 6, 0, 0)
        hlay.setSpacing(0)
        lbl_h = QLabel("Ganci  ·  click per dettagli")
        lbl_h.setStyleSheet("color: #6B7280; font-size: 10px; font-weight: 600; "
                            "text-transform: uppercase; letter-spacing: 1px; padding: 6px 12px 2px 12px;")
        hlay.addWidget(lbl_h)
        self.hooks_list = QListWidget(hooks_wrap)
        self.hooks_list.setStyleSheet("""
            QListWidget { background-color: transparent; border: none; padding: 4px 8px; }
            QListWidget::item {
                color: #93C5FD; padding: 6px 8px; border-radius: 6px;
                border: 1px solid rgba(59,130,246,30); margin: 2px 0;
            }
            QListWidget::item:hover { background-color: rgba(59,130,246,30); color: #fff; }
            QListWidget::item:selected { background-color: rgba(59,130,246,80); color: #fff; }
        """)
        self.hooks_list.itemClicked.connect(self._on_hook_clicked)
        hlay.addWidget(self.hooks_list, 1)
        splitter.addWidget(hooks_wrap)

        # Pannello suggerimenti (Fase 4: composer output) + thumbs feedback (Fase 10)
        sugg_wrap = QFrame(container)
        slay = QVBoxLayout(sugg_wrap)
        slay.setContentsMargins(0, 6, 0, 0)
        slay.setSpacing(0)
        sugg_header = QHBoxLayout()
        sugg_header.setContentsMargins(12, 6, 12, 2)
        lbl_s = QLabel("Suggerimenti LLM")
        lbl_s.setStyleSheet("color: #6B7280; font-size: 10px; font-weight: 600; "
                            "text-transform: uppercase; letter-spacing: 1px;")
        sugg_header.addWidget(lbl_s)
        sugg_header.addStretch()
        # Bottoni 👍 / 👎
        self.btn_thumbs_up = QPushButton("👍")
        self.btn_thumbs_up.setToolTip("Buono — il sistema imparerà a triggherare di più così")
        self.btn_thumbs_up.setStyleSheet(
            "background:transparent; color:#9CA3AF; border:none; padding:2px 6px;"
        )
        self.btn_thumbs_up.clicked.connect(lambda: self._on_thumbs(True))
        sugg_header.addWidget(self.btn_thumbs_up)
        self.btn_thumbs_dn = QPushButton("👎")
        self.btn_thumbs_dn.setToolTip("Non utile — il sistema imparerà a triggherare meno così")
        self.btn_thumbs_dn.setStyleSheet(
            "background:transparent; color:#9CA3AF; border:none; padding:2px 6px;"
        )
        self.btn_thumbs_dn.clicked.connect(lambda: self._on_thumbs(False))
        sugg_header.addWidget(self.btn_thumbs_dn)
        slay.addLayout(sugg_header)

        # ──────── Split LEFT (storico) / RIGHT (current) ────────
        # Vede sugg N-1 a sinistra e sugg N a destra, senza perdere il primo.
        # Quando un nuovo sugg arriva, current scorre a sinistra (= prev).
        sugg_split = QHBoxLayout()
        sugg_split.setContentsMargins(10, 4, 10, 0)
        sugg_split.setSpacing(8)

        # State pin (modificato da bottoni). Default unpinned.
        self._prev_pinned = False
        self._curr_pinned = False

        # LEFT — suggerimento precedente (storico, sbiadito)
        prev_wrap = QFrame(sugg_wrap)
        prev_wrap_lay = QVBoxLayout(prev_wrap)
        prev_wrap_lay.setContentsMargins(0, 0, 0, 0)
        prev_wrap_lay.setSpacing(2)
        prev_header = QHBoxLayout()
        prev_header.setContentsMargins(0, 0, 0, 0)
        prev_header.setSpacing(4)
        self.lbl_prev = QLabel("◀ Precedente")
        self.lbl_prev.setStyleSheet(
            "color: #6B7280; font-size: 9px; font-weight: 600; text-transform: uppercase;"
        )
        prev_header.addWidget(self.lbl_prev)
        prev_header.addStretch()
        self.btn_pin_prev = QPushButton("📌")
        self.btn_pin_prev.setCheckable(True)
        self.btn_pin_prev.setToolTip("Pin: blocca il pannello sinistro (no overwrite)")
        self.btn_pin_prev.setStyleSheet(
            "QPushButton { background:transparent; color:#6B7280; border:none; "
            "padding:0 4px; font-size:11px; } "
            "QPushButton:checked { color:#F59E0B; }"
        )
        self.btn_pin_prev.toggled.connect(self._on_toggle_pin_prev)
        prev_header.addWidget(self.btn_pin_prev)
        prev_wrap_lay.addLayout(prev_header)
        self.prev_suggestion_view = QTextEdit(prev_wrap)
        self.prev_suggestion_view.setObjectName("suggestions_prev")
        self.prev_suggestion_view.setReadOnly(True)
        # Stile "sbiadito" per il prev: opacity più bassa, font 15
        self.prev_suggestion_view.setStyleSheet(
            "QTextEdit#suggestions_prev {"
            "  background-color: rgba(255,255,255,4); color: #9CA3AF; border: none;"
            "  padding: 6px 10px 10px 10px; font-size: 11px; line-height: 1.4;"
            "}"
        )
        self.prev_suggestion_view.setPlaceholderText("(nessun suggerimento precedente)")
        prev_wrap_lay.addWidget(self.prev_suggestion_view, 1)

        # RIGHT — suggerimento corrente (acceso, principale)
        curr_wrap = QFrame(sugg_wrap)
        curr_wrap_lay = QVBoxLayout(curr_wrap)
        curr_wrap_lay.setContentsMargins(0, 0, 0, 0)
        curr_wrap_lay.setSpacing(2)
        curr_header = QHBoxLayout()
        curr_header.setContentsMargins(0, 0, 0, 0)
        curr_header.setSpacing(4)
        self.lbl_curr = QLabel("▶ Corrente")
        self.lbl_curr.setStyleSheet(
            "color: #10B981; font-size: 9px; font-weight: 700; text-transform: uppercase;"
        )
        curr_header.addWidget(self.lbl_curr)
        curr_header.addStretch()
        self.btn_pin_curr = QPushButton("📌")
        self.btn_pin_curr.setCheckable(True)
        self.btn_pin_curr.setToolTip("Pin: pausa trigger automatici (chat resta attiva)")
        self.btn_pin_curr.setStyleSheet(
            "QPushButton { background:transparent; color:#6B7280; border:none; "
            "padding:0 4px; font-size:11px; } "
            "QPushButton:checked { color:#F59E0B; }"
        )
        self.btn_pin_curr.toggled.connect(self._on_toggle_pin_curr)
        curr_header.addWidget(self.btn_pin_curr)
        curr_wrap_lay.addLayout(curr_header)
        self.suggestions_view = QTextEdit(curr_wrap)
        self.suggestions_view.setObjectName("suggestions")
        self.suggestions_view.setReadOnly(True)
        # Stile acceso: il pannello principale di lettura durante la call.
        self.suggestions_view.setStyleSheet(
            "QTextEdit#suggestions {"
            "  background-color: rgba(255,255,255,8); color: #F3F4F6; border: none;"
            "  padding: 8px 12px 12px 12px; font-size: 12px; line-height: 1.45;"
            "}"
        )
        self.suggestions_view.setPlaceholderText(
            "I suggerimenti compariranno qui durante la call "
            "(grounded sulla KB attiva + cloud LLM)."
        )
        curr_wrap_lay.addWidget(self.suggestions_view, 1)

        sugg_split.addWidget(prev_wrap, 1)   # 50% spazio
        sugg_split.addWidget(curr_wrap, 1)   # 50% spazio
        slay.addLayout(sugg_split, 1)

        # ──────── Chat input (Fase 4.2: chat mode) ────────
        chat_bar = QHBoxLayout()
        chat_bar.setContentsMargins(10, 6, 10, 4)
        chat_bar.setSpacing(6)
        self.chat_input = QLineEdit(sugg_wrap)
        self.chat_input.setPlaceholderText(
            "💬 Chiedi al copilot (es: 'fammi vedere il pricing', 'risposta migliore?')"
        )
        self.chat_input.setStyleSheet(
            "QLineEdit { background-color: rgba(255,255,255,10); color: #F3F4F6; "
            "border: 1px solid rgba(255,255,255,15); border-radius: 8px; "
            "padding: 8px 12px; font-size: 14px; } "
            "QLineEdit:focus { border-color: #10B981; }"
        )
        self.chat_input.returnPressed.connect(self._on_chat_submit)
        chat_bar.addWidget(self.chat_input, 1)
        self.btn_chat_send = QPushButton("↩")
        self.btn_chat_send.setToolTip("Invia (oppure Enter)")
        self.btn_chat_send.setStyleSheet(
            "background:#10B981; color:white; border:none; border-radius:8px; "
            "padding:8px 14px; font-weight:600;"
        )
        self.btn_chat_send.clicked.connect(self._on_chat_submit)
        chat_bar.addWidget(self.btn_chat_send)
        slay.addLayout(chat_bar)

        splitter.addWidget(sugg_wrap)

        # Pannello note (Fase 5): editable, persiste su Session
        notes_wrap = QFrame(container)
        nlay = QVBoxLayout(notes_wrap)
        nlay.setContentsMargins(0, 6, 0, 0)
        nlay.setSpacing(0)
        lbl_n = QLabel("Note  ·  appunti personali (auto-save)")
        lbl_n.setStyleSheet("color: #6B7280; font-size: 10px; font-weight: 600; "
                            "text-transform: uppercase; letter-spacing: 1px; padding: 6px 12px 2px 12px;")
        nlay.addWidget(lbl_n)
        self.notes_view = QTextEdit(notes_wrap)
        self.notes_view.setPlaceholderText(
            "Scrivi qui appunti personali durante la call.\n"
            "Salvati automaticamente nella sessione e usati per il riepilogo post-call."
        )
        self.notes_view.setStyleSheet(
            "background-color: rgba(255,255,255,5); color: #E5E7EB; border: none; "
            "padding: 8px 14px 14px 14px; font-size: 16px; line-height: 1.5;"
        )
        # Auto-save: ogni textChanged aggiorna Session
        self.notes_view.textChanged.connect(self._on_notes_changed)
        nlay.addWidget(self.notes_view, 1)
        splitter.addWidget(notes_wrap)

        # Proporzioni iniziali: trascrizione, ganci, SUGGERIMENTI (principale), notes
        # Suggerimenti = ~70% verticale per dare spazio a 2 colonne + chat input.
        # Trascrizione + ganci ridotti perché sono "scorrevoli" — chi guarda l'overlay
        # legge i suggerimenti e digita nel chat input.
        splitter.setSizes([160, 40, 640, 50])
        # Stretch factors: solo suggerimenti cresce se l'utente resize la window
        splitter.setStretchFactor(0, 0)   # trascrizione fissa
        splitter.setStretchFactor(1, 0)   # ganci fisso
        splitter.setStretchFactor(2, 1)   # suggerimenti elastico
        splitter.setStretchFactor(3, 0)   # notes fisso
        inner.addWidget(splitter, 1)

        # ── Wire dei segnali dal bridge ──
        # Qt.QueuedConnection esplicita: thread-safe, l'update arriva sull'event loop UI.
        self.bridge.new_line.connect(self.on_new_line, Qt.ConnectionType.QueuedConnection)
        self.bridge.status.connect(self.on_status, Qt.ConnectionType.QueuedConnection)
        # Segnali LLM dall'orchestrator
        self.bridge.llm_status.connect(self._on_llm_status, Qt.ConnectionType.QueuedConnection)
        self.bridge.llm_token.connect(self._on_llm_token, Qt.ConnectionType.QueuedConnection)
        self.bridge.llm_complete.connect(self._on_llm_complete, Qt.ConnectionType.QueuedConnection)
        # Segnali Watcher (Fase 6)
        self.bridge.watcher_decision.connect(self._on_watcher_decision, Qt.ConnectionType.QueuedConnection)
        self.bridge.watcher_token.connect(self._on_watcher_token, Qt.ConnectionType.QueuedConnection)
        self.bridge.watcher_complete.connect(self._on_watcher_complete, Qt.ConnectionType.QueuedConnection)

        # Esc → chiudi
        QShortcut(QKeySequence("Escape"), self).activated.connect(self.close)
        # Cmd+Shift+T → toggle visibilità (funziona solo quando l'app ha focus)
        QShortcut(QKeySequence("Ctrl+Shift+T"), self).activated.connect(self._toggle_visibility)
        # Cmd+Shift+R → retrieval RAG sulle ultime parole della trascrizione
        QShortcut(QKeySequence("Ctrl+Shift+R"), self).activated.connect(self.trigger_rag_search)
        # Cmd+Shift+S → salva ultime righe come chunk in una sezione KB
        QShortcut(QKeySequence("Ctrl+Shift+S"), self).activated.connect(self.save_passage_to_kb)
        # Cmd+Shift+N → focus sul pannello note
        QShortcut(QKeySequence("Ctrl+Shift+N"), self).activated.connect(
            lambda: self.notes_view.setFocus()
        )

        # Aggiorna label KB iniziale
        self._update_kb_label()

        # Inizializza il Watcher e attiva auto-mode di default (config-driven)
        from .config import WATCHER as _WATCHER_CFG
        self._ensure_watcher()
        if _WATCHER_CFG.default_enabled:
            self.header.btn_auto.setChecked(True)
            self.header.btn_auto.setText("◆ Auto")

        # Global hotkeys (Fase 5): funzionano anche con Meet in foreground.
        # ⚠️ Richiede permission Accessibility (System Settings → Privacy & Security).
        self._global_hotkeys = GlobalHotkeys()
        # Thread-safety: i callback NSEvent vengono chiamati su main thread Cocoa,
        # ma le chiamate Qt vanno fatte tramite signal. Usiamo invokeMethod via metodo bridge.
        self._global_hotkeys.register("R", self._invoke_trigger_rag)
        self._global_hotkeys.register("S", self._invoke_save_passage)
        self._global_hotkeys.register("N", self._invoke_focus_notes)
        self._global_hotkeys.register("T", self._invoke_toggle_visibility)
        self._global_hotkeys.register("B", self._invoke_cloud_burst)  # Fase 18+19
        self._global_hotkeys.start()

    # Bridge globali: chiamate da PyObjC handler thread. Usiamo singleShot
    # per delegare al main thread Qt safely.
    def _invoke_trigger_rag(self) -> None:
        from PyQt6.QtCore import QTimer as _QT
        _QT.singleShot(0, self.trigger_rag_search)

    def _invoke_save_passage(self) -> None:
        from PyQt6.QtCore import QTimer as _QT
        _QT.singleShot(0, self.save_passage_to_kb)

    def _invoke_focus_notes(self) -> None:
        from PyQt6.QtCore import QTimer as _QT
        _QT.singleShot(0, lambda: (self.show(), self.raise_(), self.notes_view.setFocus()))

    def _invoke_toggle_visibility(self) -> None:
        from PyQt6.QtCore import QTimer as _QT
        _QT.singleShot(0, self._toggle_visibility)

    def _invoke_cloud_burst(self) -> None:
        """Fase 18+19 — Cmd+Shift+B: chiama cloud LLM premium con full context.

        Esegue in thread separato (l'HTTP è bloccante e non vogliamo freezare l'UI).
        Stream dei token via signal Qt-safe.
        """
        from PyQt6.QtCore import QTimer as _QT
        _QT.singleShot(0, self._launch_cloud_burst)

    def _launch_cloud_burst(self) -> None:
        from .config import CLOUD_BURST as _CB
        if not _CB.enabled:
            self.header.set_status(
                "● Cloud Burst disattivato — vedi config.CLOUD_BURST",
                color="#EF4444",
            )
            return

        # Estrai query: ultima riga "rilevante" della trascrizione (non vuota).
        query = ""
        for line in reversed(self._recent_lines[-6:]):
            if line and line.strip():
                query = line.strip()
                break
        if not query:
            self.header.set_status("● Cloud Burst: nessuna query", color="#F59E0B")
            return

        # Status: badge "cloud"
        self.header.set_status(f"● ☁️ Cloud Burst ({_CB.model})…", color="#8B5CF6")

        # Recupera chunks KB attivi + memorie correlate
        try:
            from .kb.manager import KnowledgeManager
            active_sections = KnowledgeManager.get_active_sections()
            kb_chunks = []
            if active_sections:
                kb_chunks = KnowledgeManager.search(query, sections=active_sections, top_k=6)
        except Exception as e:
            print(f"[cloud_burst] KB error: {e}")
            kb_chunks = []
            active_sections = []

        try:
            from .memory import MemoryManager
            from .config import MEMORY as _MEM
            mem_texts: list[str] = []
            if _MEM.enabled and active_sections:
                mems = MemoryManager.search(query, active_sections, top_k=5)
                mem_texts = [r.text for r, _s in mems]
        except Exception as e:
            print(f"[cloud_burst] memory error: {e}")
            mem_texts = []

        # Thread per non bloccare l'UI durante streaming
        import threading
        from PyQt6.QtCore import QTimer as _QT

        burst_id = f"burst-{int(__import__('time').time())}"

        def append_token(piece: str) -> None:
            # Qt-safe via singleShot
            _QT.singleShot(0, lambda p=piece: self.notes_view.append(p) if hasattr(self, "notes_view") else None)

        def worker():
            try:
                from .llm.cloud_burst import cloud_burst
                _QT.singleShot(0, lambda: self.notes_view.append(
                    f"\n\n<b style='color:#8B5CF6;'>☁️ Cloud Burst — {_CB.model}</b>\n"
                ) if hasattr(self, "notes_view") else None)
                full = cloud_burst(
                    query=query,
                    transcript_lines=list(self._recent_lines),
                    kb_chunks=kb_chunks,
                    memories=mem_texts,
                    on_token=append_token,
                )
                _QT.singleShot(0, lambda: self.header.set_status(
                    f"● ☁️ Cloud Burst done ({len(full)} chars)", color="#10B981"
                ))
            except Exception as e:
                print(f"[cloud_burst] error: {e}")
                _QT.singleShot(0, lambda err=e: self.header.set_status(
                    f"● ☁️ Cloud Burst error: {err}", color="#EF4444"
                ))

        threading.Thread(target=worker, daemon=True, name=burst_id).start()

    # ── slot ──
    @pyqtSlot(str, str, float)
    def on_new_line(self, text: str, language: str, duration_s: float, speaker: str = "unknown") -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        lang_tag = f"<span style='color:#6B7280; font-size:11px;'>[{language}]</span> " if language else ""
        line = (
            f"<div style='margin-bottom:6px;'>"
            f"<span style='color:#4B5563; font-size:11px;'>{ts}</span> "
            f"{lang_tag}"
            f"<span style='color:#E5E7EB;'>{_escape_html(text)}</span>"
            f"</div>"
        )
        self.transcript_view.append(line)
        # Autoscroll
        bar = self.transcript_view.verticalScrollBar()
        bar.setValue(bar.maximum())
        # Aggiungi al buffer recente per il RAG retrieval, con speaker label
        # in modo che il composer sappia distinguere TU vs ALTRO partecipante.
        speaker_prefix = {
            "self": "[TU]",
            "other": "[ALTRO]",
            "both": "[ENTRAMBI]",
        }.get(speaker, "[??]")
        labeled = f"{speaker_prefix} {text}"
        self._recent_lines.append(labeled)
        # Persisti nella sessione corrente
        try:
            self._session.add_transcript_line(text, language=language, duration_s=duration_s)
        except Exception:
            pass
        # ⚡ Fast-path trigger (LLM.fast_path_enabled): se la nuova riga ha
        # un pattern forte di domanda, lancia composer immediato bypassando
        # il wait di auto_interval_s.
        # Pin: se curr_pinned, blocca i trigger automatici (chat resta attiva).
        try:
            if getattr(self, "_curr_pinned", False):
                pass  # in pausa, no trigger
            elif hasattr(self, "_orchestrator") and self._orchestrator is not None:
                self._orchestrator.consider_line(text)
        except Exception:
            pass  # silenzioso, non vogliamo che fast-path rompa l'UI

    @pyqtSlot(str)
    def on_status(self, msg: str) -> None:
        self.header.set_status(f"● {msg}", color="#F59E0B")

    # ── Session lifecycle ──
    def close_session_and_summarize(self) -> str | None:
        """Chiude la session corrente e genera il markdown di sintesi.
        Ritorna il path del file di sintesi, o None se non generato."""
        try:
            self._session.close()
            # Stop hotkeys
            try:
                self._global_hotkeys.stop()
            except Exception:
                pass
            # Solo se c'è materiale e l'utente non ha annullato
            if not self._session.transcript_lines and not self._session.notes:
                print(f"[session] sessione vuota, skipping summary.")
                return None
            print("[session] generating post-call summary…")
            from .llm.summarizer import summarize_session
            summary_path = summarize_session(self._session)
            self._session.set_summary_path(summary_path)
            print(f"[session] ✓ summary saved: {summary_path}")
            return str(summary_path)
        except Exception as e:
            print(f"[session] errore: {e}")
            return None

    # ── public actions ──
    def copy_transcript(self) -> None:
        QApplication.clipboard().setText(self.transcript_view.toPlainText())
        self.header.set_status("● Copiato negli appunti", color="#3B82F6")

    def _toggle_visibility(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()

    # ── KB menu ──
    def show_kb_menu(self) -> None:
        """Mostra un menu popup con le sezioni di knowledge disponibili
        e checkboxes per attivare/disattivare quelle attive per la call."""
        try:
            sections = KnowledgeManager.list_sections()
            active = set(KnowledgeManager.get_active_sections())
        except Exception as e:
            self.header.set_status(f"KB errore: {e}", color="#EF4444")
            return

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background-color: #15171C; color: #E5E7EB; border: 1px solid #2A2D35;
                    border-radius: 8px; padding: 6px; }
            QMenu::item { padding: 6px 14px; border-radius: 4px; }
            QMenu::item:selected { background-color: #1F2937; }
        """)

        if not sections:
            act = menu.addAction("Nessuna sezione. Crea con CLI o Knowledge Manager.")
            act.setEnabled(False)
        else:
            for s in sections:
                cb = QCheckBox(f"{s.slug}  ({s.n_chunks} chunks)")
                cb.setChecked(s.slug in active)
                cb.setStyleSheet("padding: 6px 14px; color: #E5E7EB;")
                # Closure su slug
                cb.stateChanged.connect(
                    lambda state, slug=s.slug: self._toggle_kb_section(slug, state)
                )
                wa = QWidgetAction(menu)
                wa.setDefaultWidget(cb)
                menu.addAction(wa)

        menu.addSeparator()
        act_open = menu.addAction("📁  Apri Knowledge Manager…")
        act_open.triggered.connect(self._open_knowledge_manager)

        # Popup sotto il bottone
        global_pos = self.header.btn_kb.mapToGlobal(
            self.header.btn_kb.rect().bottomLeft()
        )
        menu.exec(global_pos)

    def _toggle_kb_section(self, slug: str, state) -> None:
        active = KnowledgeManager.get_active_sections()
        is_checked = state == Qt.CheckState.Checked.value
        if is_checked and slug not in active:
            active.append(slug)
        elif not is_checked and slug in active:
            active.remove(slug)
        KnowledgeManager.set_active_sections(active)
        self._update_kb_label()

    def _update_kb_label(self) -> None:
        try:
            n = len(KnowledgeManager.get_active_sections())
        except Exception:
            n = 0
        self.header.btn_kb.setText(f"📚 KB ({n})" if n else "📚 KB")

    def _open_knowledge_manager(self) -> None:
        """Apre la finestra Knowledge Manager. Lazy import per evitare cicli."""
        from .knowledge_ui import KnowledgeManagerWindow
        # Tieni un riferimento per evitare GC
        if not hasattr(self, "_kb_window") or self._kb_window is None:
            self._kb_window = KnowledgeManagerWindow()
        self._kb_window.show()
        self._kb_window.raise_()
        self._kb_window.activateWindow()

    def open_settings(self) -> None:
        """Apre la finestra Settings (Fase 11)."""
        from .ui.settings import open_settings as _open_settings
        _open_settings(parent=self)

    # ── Watcher Agent (Fase 6) ──
    def _ensure_watcher(self) -> WatcherAgent:
        """Lazy init del Watcher."""
        if self._watcher is None:
            self._watcher = WatcherAgent(
                on_decision=lambda d: self.bridge.watcher_decision.emit(d),
                on_token=lambda did, piece: self.bridge.watcher_token.emit(did, piece),
                on_complete=lambda did, txt, hits: self.bridge.watcher_complete.emit(did, txt, hits),
                on_status=lambda txt, color: self.bridge.llm_status.emit(txt, color),
            )
        return self._watcher

    def on_new_transcript_for_watcher(self, text: str, speaker_label: str) -> None:
        """Chiamata DAL thread STT quando arriva una nuova riga.
        Inoltra al Watcher per valutazione (Watcher è thread-safe internamente)."""
        try:
            watcher = self._ensure_watcher()
            watcher.consider(list(self._recent_lines), speaker_label)
        except Exception as e:
            print(f"[overlay] watcher.consider error: {e}")

    @pyqtSlot(object)
    def _on_watcher_decision(self, decision: WatcherDecision) -> None:
        """Slot: ogni decisione del Watcher aggiorna UI tier-based."""
        # 1) Aggiorna l'header con lo score + grounding emoji (Fase 13)
        color = self._score_to_color(decision.score, decision.tier)
        grounding_emoji = {
            "high": "📎",     # KB strong
            "medium": "📎🌐", # KB partial + LLM
            "low": "🌐",     # LLM only
        }.get(decision.grounding_tier, "")
        status_text = f"● {decision.score:.2f} · {decision.tier} {grounding_emoji}"
        if decision.speaker == "self":
            status_text = f"○ self  · {decision.score:.2f}"
            color = "#6B7280"
        self.header.set_status(status_text, color=color)

        # 2) Tier-specific behavior
        if decision.tier in ("silent", "indicator"):
            return  # niente UI updates

        if decision.tier == "hook":
            # Aggiungi all'elenco ganci (se non già presente)
            if decision.id not in self._active_hooks:
                self._active_hooks[decision.id] = decision.hook_label
                item = QListWidgetItem(decision.hook_label or "💡 (gancio)")
                item.setData(Qt.ItemDataRole.UserRole, decision.id)
                self.hooks_list.insertItem(0, item)  # più recenti in cima
                # Telemetry: hook mostrato
                try:
                    from .telemetry import record_hook_shown
                    record_hook_shown(decision.id)
                except Exception:
                    pass
                # Mantieni max 8 ganci
                while self.hooks_list.count() > 8:
                    last = self.hooks_list.takeItem(self.hooks_list.count() - 1)
                    if last:
                        rid = last.data(Qt.ItemDataRole.UserRole)
                        self._active_hooks.pop(rid, None)
                        # Telemetry: hook dismissed (scaduto senza click)
                        try:
                            from .telemetry import record_hook_dismissed
                            record_hook_dismissed(rid)
                        except Exception:
                            pass
            return

        if decision.tier == "auto":
            # AUTO-FIRE: pannello suggerimenti show this decision
            self._current_displayed_decision = decision.id
            self._composer_buffer.clear()
            self.suggestions_view.setHtml(
                f"<p style='color:#9CA3AF;'><i>💡 Risposta auto su: "
                f"{_escape_html(decision.hook_label[:80])}</i></p>"
            )
            # Persisti nella session
            self._last_query = decision.query
            self._last_category = "auto"
            self._last_confidence = decision.score
            # Telemetry: hook "auto" mostrato
            try:
                from .telemetry import record_hook_shown
                record_hook_shown(decision.id)
            except Exception:
                pass
            return

    @pyqtSlot(str, str)
    def _on_watcher_token(self, decision_id: str, piece: str) -> None:
        """Streaming token dal composer speculativo del Watcher.
        Aggiorna il pannello suggerimenti SE è la decisione attualmente visualizzata."""
        if decision_id != self._current_displayed_decision:
            return  # token in background per un gancio non ancora cliccato
        self._composer_buffer.append(piece)
        self._stream_dirty = True
        if not self._stream_timer.isActive():
            self._stream_timer.start()

    @pyqtSlot(str, str, list)
    def _on_watcher_complete(self, decision_id: str, full_text: str, hits: list) -> None:
        """Fine streaming del Watcher. Se è la decisione visualizzata,
        renderizza finale; salvalo sempre nella session."""
        if decision_id == self._current_displayed_decision:
            self._stream_timer.stop()
            self._composer_buffer = [full_text]
            html = _md_to_html(full_text)
            if hits:
                sources_seen: list[str] = []
                for h in hits:
                    tag = f"{h.section} · {h.source}"
                    if tag not in sources_seen:
                        sources_seen.append(tag)
                html += (
                    "<hr style='border:none; border-top:1px solid #2A2D35; margin:10px 0;'>"
                    "<div style='color:#6B7280; font-size:11px;'>"
                    "📎 Fonti: " + ", ".join(_escape_html(s) for s in sources_seen) +
                    "</div>"
                )
            self.suggestions_view.setHtml(html)

        # Persisti sempre nella session
        try:
            sources_tags = [f"{h.section} · {h.source}" for h in hits] if hits else []
            self._session.add_suggestion(
                category=self._last_category or "auto",
                query=self._last_query,
                text=full_text,
                sources=sources_tags,
                confidence=self._last_confidence,
            )
        except Exception:
            pass

    def _on_hook_clicked(self, item) -> None:
        """Click su un gancio: cerca il risultato speculativo cached e mostra."""
        decision_id = item.data(Qt.ItemDataRole.UserRole)
        if not decision_id:
            return
        # Telemetry: record click
        try:
            from .telemetry import record_hook_clicked
            record_hook_clicked(decision_id)
        except Exception:
            pass
        self._current_displayed_decision = decision_id
        watcher = self._ensure_watcher()
        spec = watcher.user_clicked_hook(decision_id)
        if spec is None:
            self.suggestions_view.setHtml(
                "<p style='color:#9CA3AF;'>"
                "<i>Gancio cliccato ma risultato scaduto. Premi 💡 Ask per rigenerare.</i></p>"
            )
            return
        # Se ancora in progress, mostra parziale + il timer dello streaming aggiornerà
        if spec.text and not spec.done:
            self._composer_buffer = [spec.text]
            html = _md_to_html(spec.text)
            self.suggestions_view.setHtml(html + "<p style='color:#6B7280;'><i>… generazione in corso</i></p>")
        elif spec.done:
            self._composer_buffer = [spec.text]
            html = _md_to_html(spec.text)
            if spec.hits:
                sources_seen = []
                for h in spec.hits:
                    tag = f"{h.section} · {h.source}"
                    if tag not in sources_seen:
                        sources_seen.append(tag)
                html += (
                    "<hr style='border:none; border-top:1px solid #2A2D35; margin:10px 0;'>"
                    "<div style='color:#6B7280; font-size:11px;'>"
                    "📎 Fonti: " + ", ".join(_escape_html(s) for s in sources_seen) +
                    "</div>"
                )
            self.suggestions_view.setHtml(html)
        else:
            self.suggestions_view.setHtml(
                "<p style='color:#9CA3AF;'><i>⚙️ Generazione in corso…</i></p>"
            )

    def _on_thumbs(self, positive: bool) -> None:
        """Fase 10: utente vota il suggerimento corrente. Il sistema impara."""
        if not self._current_displayed_decision:
            return
        try:
            from .telemetry import record_thumbs
            record_thumbs(self._current_displayed_decision, positive)
        except Exception:
            pass
        # Feedback visivo veloce
        sign = "👍" if positive else "👎"
        self.header.set_status(f"● Feedback {sign} registrato", color="#10B981" if positive else "#F59E0B")

    @staticmethod
    def _score_to_color(score: float, tier: str) -> str:
        if tier == "silent" or tier == "indicator":
            return "#6B7280"  # grey
        if tier == "hook":
            return "#3B82F6"  # blue
        if tier == "auto":
            return "#10B981"  # green
        return "#9CA3AF"

    # ── LLM orchestrator (Fase 4) ──
    def _ensure_orchestrator(self) -> Orchestrator:
        """Inizializza l'orchestrator lazy alla prima richiesta."""
        if self._orchestrator is None:
            self._orchestrator = Orchestrator(
                get_recent_lines=lambda: list(self._recent_lines)[-10:],
                on_status=lambda txt, color: self.bridge.llm_status.emit(txt, color),
                on_analysis=self._on_analysis,
                on_token=lambda piece: self.bridge.llm_token.emit(piece),
                on_complete=lambda txt, hits: self.bridge.llm_complete.emit(txt, hits),
            )
            self._orchestrator.start()
        return self._orchestrator

    def _on_analysis(self, analysis) -> None:
        """Memorizza categoria/query/confidence dell'ultima analisi
        per loggarli nella session quando arriva il composer complete."""
        self._last_query = analysis.query
        self._last_category = analysis.category
        self._last_confidence = analysis.confidence

    # ── Notes panel ──
    def _on_notes_changed(self) -> None:
        """Auto-save delle note nella session ad ogni edit."""
        try:
            text = self.notes_view.toPlainText()
            self._session.update_notes(text)
        except Exception:
            pass

    # ── Save passage to KB (Cmd+Shift+S) ──
    def save_passage_to_kb(self) -> None:
        """Salva le ultime ~3 righe della trascrizione come chunk in una
        sezione KB esistente. Chiede all'utente quale sezione."""
        if not self._recent_lines:
            QMessageBox.information(
                self, "Save passage",
                "Niente trascrizione recente da salvare."
            )
            return

        sections = KnowledgeManager.list_sections()
        if not sections:
            QMessageBox.warning(
                self, "Nessuna sezione",
                "Non hai sezioni KB. Crea prima una sezione con il Knowledge Manager."
            )
            return

        # Costruisce il passaggio dalle ultime righe
        passage = "\n".join(list(self._recent_lines)[-5:]).strip()
        if not passage:
            return

        # Dialog: scelta sezione
        slugs = [s.slug for s in sections]
        slug, ok = QInputDialog.getItem(
            self, "Salva passaggio in KB",
            f"Scegli la sezione dove salvare questo chunk:\n\n"
            f"{passage[:200]}{'…' if len(passage) > 200 else ''}",
            slugs,
            0,
            False,
        )
        if not ok or not slug:
            return

        # Crea un file .md nella cartella sources/ della sezione
        from .kb import paths as kb_paths
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        src_dir = kb_paths.sources_dir(slug)
        src_dir.mkdir(parents=True, exist_ok=True)
        out_file = src_dir / f"passage_{ts}.md"
        out_file.write_text(
            f"# Passaggio salvato in call\n\n"
            f"_Sessione: {self._session.id} · Salvato il: {ts}_\n\n"
            f"{passage}\n",
            encoding="utf-8",
        )

        # Reindex in background
        self.header.set_status(f"● Saving to {slug}…", color="#3B82F6")
        try:
            from .kb.manager import KnowledgeManager as KM
            KM.reindex(slug, verbose=False)
        except Exception as e:
            QMessageBox.critical(self, "Errore reindex", str(e))
            return

        # Registra nella sessione
        self._session.add_saved_passage(slug, passage)
        self.header.set_status(f"● Salvato in '{slug}' ✓", color="#10B981")

    def trigger_rag_search(self) -> None:
        """Cmd+Shift+R: triggera Analyzer → Retriever → Composer."""
        active = KnowledgeManager.get_active_sections()
        if not active:
            self.suggestions_view.setHtml(
                "<p style='color:#F59E0B;'>"
                "Nessuna sezione di knowledge attiva. "
                "Click su <b>📚 KB</b> in alto per attivarne.</p>"
            )
            return
        if not self._recent_lines:
            self.suggestions_view.setHtml(
                "<p style='color:#9CA3AF;'>Niente trascrizione recente su cui cercare.</p>"
            )
            return

        orch = self._ensure_orchestrator()
        # Pulisci pannello suggerimenti e prepara buffer streaming
        self._composer_buffer = []
        self.suggestions_view.setHtml(
            "<p style='color:#9CA3AF;'><i>⚙️ Triggering…</i></p>"
        )
        orch.request_now()

    def set_auto_mode(self, enabled: bool) -> None:
        """Toggle Auto/Manual dall'header. In Fase 6 controlla il Watcher.
        Quando disabilitato, il Watcher rimane silenzioso (no auto-trigger,
        no hooks). Il bottone 💡 Ask resta comunque attivo."""
        watcher = self._ensure_watcher()
        watcher.set_enabled(enabled)
        self.header.btn_auto.setText("◆ Auto" if enabled else "◇ Auto")
        if enabled:
            self.header.set_status("● Watcher ON", color="#10B981")
        else:
            self.header.set_status("● Watcher OFF — solo manuale", color="#6B7280")
            # Pulisce hooks pending
            self.hooks_list.clear()
            self._active_hooks.clear()

    @pyqtSlot(str, str)
    def _on_llm_status(self, text: str, color: str) -> None:
        self.header.set_status(text, color=color)
        # UI #2: quando inizia un nuovo "Generating", sposta il sugg corrente
        # nel pannello "Precedente" prima di accogliere il nuovo stream.
        # Eccezione: se il current è il placeholder chat "💬 Chiesto: ..."
        # (già abbiamo shiftato in _on_chat_submit) → solo clear, no double-shift.
        if "Generating" in text:
            current_text = self.suggestions_view.toPlainText().strip()
            if current_text.startswith("💬"):
                # Chat placeholder già piazzato: clear e via, no shift
                self.suggestions_view.clear()
                self._composer_buffer.clear()
            else:
                self._shift_suggestion_to_prev()

    def _shift_suggestion_to_prev(self) -> None:
        """Sposta il suggerimento corrente nel pannello sinistro 'Precedente'.

        Pin behavior:
        - Se prev_pinned: NON sovrascrive prev. Current si pulisce normalmente.
        - Se curr_pinned: questa chiamata non dovrebbe accadere (i trigger
          dovrebbero essere bloccati upstream). Defensive: skip totale.
        """
        try:
            if self._curr_pinned:
                # Defensive: se per qualche motivo siamo qui con curr pinned,
                # NON tocchiamo niente (current resta dov'è).
                return
            current_text = self.suggestions_view.toPlainText().strip()
            if not current_text:
                return
            if self._prev_pinned:
                # Prev locked: scarta il current vecchio, NON overwrite prev.
                self.suggestions_view.clear()
                self._composer_buffer.clear()
                return
            # Normale: shift current → prev
            self.prev_suggestion_view.setHtml(self.suggestions_view.toHtml())
            self.suggestions_view.clear()
            self._composer_buffer.clear()
        except Exception as e:
            print(f"[overlay] shift suggestion error: {e}")

    def _on_toggle_pin_prev(self, checked: bool) -> None:
        """Toggle pin sul pannello sinistro (precedente)."""
        self._prev_pinned = checked
        self.lbl_prev.setText("◀ 📌 Pinnato" if checked else "◀ Precedente")
        self.lbl_prev.setStyleSheet(
            ("color: #F59E0B" if checked else "color: #6B7280") +
            "; font-size: 9px; font-weight: 600; text-transform: uppercase;"
        )

    def _on_toggle_pin_curr(self, checked: bool) -> None:
        """Toggle pin sul pannello corrente (= pausa trigger automatici)."""
        self._curr_pinned = checked
        self.lbl_curr.setText("▶ 📌 Pausa" if checked else "▶ Corrente")
        self.lbl_curr.setStyleSheet(
            ("color: #F59E0B" if checked else "color: #10B981") +
            "; font-size: 9px; font-weight: 700; text-transform: uppercase;"
        )
        if checked:
            print("[overlay] 📌 current pinned → fast-path triggers in pausa")
        else:
            print("[overlay] 📌 current unpinned → trigger ripresi")

    def _on_chat_submit(self) -> None:
        """Chat mode: invia testo utente al orchestrator come fast-path hint.

        Riusa la pipeline esistente: skip analyzer, query = testo utente,
        category = "question". Il composer risponde con KB attive + transcript.
        """
        text = self.chat_input.text().strip()
        if not text:
            return
        self.chat_input.clear()
        self._ensure_orchestrator()
        if self._orchestrator is None:
            self.suggestions_view.setHtml(
                "<p style='color:#EF4444;'>Orchestrator non disponibile.</p>"
            )
            return
        # Mostra subito feedback visuale (placeholder mentre il composer parte)
        self._shift_suggestion_to_prev()
        self.suggestions_view.setHtml(
            f"<p style='color:#9CA3AF;'><i>💬 Chiesto: {_escape_html(text[:80])}…</i></p>"
        )
        try:
            self._orchestrator.chat_request(text)
        except AttributeError:
            # Orchestrator vecchio senza chat_request → fallback su fast_path_hint
            try:
                with self._orchestrator._lock:
                    self._orchestrator._fast_path_hint = {
                        "query": text,
                        "category": "question",
                        "confidence": 1.0,
                        "signal": "chat",
                    }
                self._orchestrator._request_event.set()
            except Exception as e:
                print(f"[overlay] chat fallback error: {e}")

    @pyqtSlot(str)
    def _on_llm_token(self, piece: str) -> None:
        """Streaming: accoda token. Il rendering avviene throttled via QTimer."""
        self._composer_buffer.append(piece)
        self._stream_dirty = True
        if not self._stream_timer.isActive():
            self._stream_timer.start()

    def _flush_stream(self) -> None:
        """Render streaming throttled: max 10 update/s anziché 1 per token."""
        if not self._stream_dirty:
            return
        self._stream_dirty = False
        full = "".join(self._composer_buffer)
        html = _md_to_html(full)
        self.suggestions_view.setHtml(html)
        # Autoscroll
        bar = self.suggestions_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    @pyqtSlot(str, list)
    def _on_llm_complete(self, full_text: str, hits: list) -> None:
        """Fine streaming: render finale + footer con fonti."""
        self._stream_timer.stop()
        # Persisti il suggerimento nella sessione
        try:
            sources_tags = [f"{h.section} · {h.source}" for h in hits] if hits else []
            self._session.add_suggestion(
                category=self._last_category or "unknown",
                query=self._last_query,
                text=full_text,
                sources=sources_tags,
                confidence=self._last_confidence,
            )
        except Exception as e:
            print(f"[overlay] errore salvataggio suggestion: {e}")
        html = _md_to_html(full_text)
        if hits:
            sources_seen: list[str] = []
            for h in hits:
                tag = f"{h.section} · {h.source}"
                if tag not in sources_seen:
                    sources_seen.append(tag)
            html += (
                "<hr style='border:none; border-top:1px solid #2A2D35; margin:10px 0;'>"
                "<div style='color:#6B7280; font-size:11px;'>"
                "📎 Fonti: " + ", ".join(_escape_html(s) for s in sources_seen) +
                "</div>"
            )
        self.suggestions_view.setHtml(html)


def _md_to_html(md: str) -> str:
    """Markdown → HTML grezzo (no librerie esterne).
    Supporta: **bold**, *italic*, paragrafi su righe vuote, line breaks."""
    import re as _re
    s = _escape_html(md)
    # Bold **text**
    s = _re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    # Italic *text* (greedy ma after bold)
    s = _re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", s)
    # Paragrafi (doppia newline)
    paragraphs = s.split("\n\n")
    body = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        p = p.replace("\n", "<br>")
        body += f"<p style='color:#E5E7EB; line-height:1.5; margin:0 0 8px 0;'>{p}</p>"
    return body

    # ── overrides ──
    def showEvent(self, ev):
        super().showEvent(ev)
        # Applica NSWindowSharingNone DOPO che la finestra è realmente sullo
        # schermo (winId() ora è valido).
        ok_share = make_invisible_to_screen_share(self)
        ok_level = set_window_always_on_top_floating(self)
        if ok_share:
            print("[overlay] ✓ NSWindowSharingNone applicato — invisibile a screen share.")
        else:
            print("[overlay] ⚠️  NSWindowSharingNone NON applicato — la finestra è visibile in screen share.")
        if ok_level:
            print("[overlay] ✓ Window level: NSFloatingWindowLevel.")


def _escape_html(s: str) -> str:
    """Escape minimo per evitare che caratteri come < o & rompano l'HTML."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
