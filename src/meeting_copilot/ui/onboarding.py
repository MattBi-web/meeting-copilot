"""Onboarding wizard al primo avvio.

Sequenza 5 step:
1. Welcome — chi siamo, cosa facciamo
2. Audio check — verifica BlackHole + MeetingCapture
3. KB setup — crea prima sezione + drop primo file
4. Hotkey cheat sheet
5. Done — sei pronto

Stato "onboarded" persistito in ~/.meeting_copilot/onboarded.json.
Skippato in run successivi. Riapribile dal menu Settings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..kb.manager import KnowledgeManager
from .theme import THEME, qss


_ONBOARDED_FILE = Path.home() / ".meeting_copilot" / "onboarded.json"


def is_onboarded() -> bool:
    return _ONBOARDED_FILE.exists()


def mark_onboarded() -> None:
    _ONBOARDED_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ONBOARDED_FILE.write_text(json.dumps({"version": 1}), encoding="utf-8")


def reset_onboarded() -> None:
    """Forza rerun dell'onboarding (utile da Settings)."""
    if _ONBOARDED_FILE.exists():
        _ONBOARDED_FILE.unlink()


# ─────────────────────────────────────────────────────────────
# Step pages — ognuna è una QWidget
# ─────────────────────────────────────────────────────────────
class _StepWelcome(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 40, 40, 40)
        lay.setSpacing(16)
        lay.addStretch()

        t = QLabel("Benvenuto in Meeting Copilot")
        t.setStyleSheet(qss.title_h1())
        lay.addWidget(t)

        sub = QLabel(
            "Il tuo assistente per videochiamate consulenziali.\n"
            "Trascrive in tempo reale, analizza la conversazione, ti suggerisce "
            "risposte basate sulla tua knowledge base — completamente in locale."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet(qss.body_dim() + "line-height:1.7;")
        lay.addWidget(sub)

        feat = QLabel(
            "<div style='margin-top:24px;'>"
            f"<p style='{qss.body()}'>✓ &nbsp;Trascrizione real-time invisibile a screen share</p>"
            f"<p style='{qss.body()}'>✓ &nbsp;Watcher Agent che capisce quando suggerire</p>"
            f"<p style='{qss.body()}'>✓ &nbsp;Knowledge base personale (PDF, DOCX, PPTX, URL)</p>"
            f"<p style='{qss.body()}'>✓ &nbsp;Zero cloud — tutto sul tuo Mac</p>"
            "</div>"
        )
        feat.setWordWrap(True)
        lay.addWidget(feat)

        lay.addStretch()


class _StepApiKeys(QWidget):
    """Step 2: API keys + server URL salvate in ~/.config/meeting_copilot/.env"""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 30, 40, 30)
        lay.setSpacing(12)

        t = QLabel("Configura le API keys")
        t.setStyleSheet(qss.title_h1())
        lay.addWidget(t)

        sub = QLabel(
            "Servono per il cloud LLM (risposte real-time) e per la trascrizione "
            "premium. Le chiavi restano sul tuo Mac (in ~/.config/meeting_copilot/.env)."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet(qss.body_dim())
        lay.addWidget(sub)

        # Anthropic
        lay.addWidget(QLabel("<b>Anthropic API Key</b> <span style='color:#9CA3AF;'>(per Claude Haiku/Sonnet)</span>"))
        self.input_anth = QLineEdit()
        self.input_anth.setPlaceholderText("sk-ant-api03-…")
        self.input_anth.setEchoMode(QLineEdit.EchoMode.Password)
        self.input_anth.setStyleSheet(qss.line_edit())
        lay.addWidget(self.input_anth)
        link_anth = QLabel("<a href='https://console.anthropic.com/settings/keys' style='color:#10B981;'>↗ ottieni key su console.anthropic.com</a>")
        link_anth.setOpenExternalLinks(True)
        link_anth.setStyleSheet("font-size:11px;")
        lay.addWidget(link_anth)

        # Deepgram
        lay.addWidget(QLabel("<b>Deepgram API Key</b> <span style='color:#9CA3AF;'>(per trascrizione real-time)</span>"))
        self.input_dg = QLineEdit()
        self.input_dg.setPlaceholderText("xxxxxxxxxxxxxxxxxxxx")
        self.input_dg.setEchoMode(QLineEdit.EchoMode.Password)
        self.input_dg.setStyleSheet(qss.line_edit())
        lay.addWidget(self.input_dg)
        link_dg = QLabel("<a href='https://console.deepgram.com' style='color:#10B981;'>↗ ottieni key su console.deepgram.com</a>")
        link_dg.setOpenExternalLinks(True)
        link_dg.setStyleSheet("font-size:11px;")
        lay.addWidget(link_dg)

        # Server (optional)
        lay.addWidget(QLabel("<b>Server URL</b> <span style='color:#9CA3AF;'>(opzionale, se hai un VM dedicato)</span>"))
        self.input_srv_url = QLineEdit()
        self.input_srv_url.setPlaceholderText("http://localhost:8787 — lascia vuoto per non usare server")
        self.input_srv_url.setStyleSheet(qss.line_edit())
        lay.addWidget(self.input_srv_url)

        self.input_srv_key = QLineEdit()
        self.input_srv_key.setPlaceholderText("Bearer token server (se URL impostato)")
        self.input_srv_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.input_srv_key.setStyleSheet(qss.line_edit())
        lay.addWidget(self.input_srv_key)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("font-size:12px; margin-top:8px;")
        lay.addWidget(self.lbl_status)

        # Pre-fill se già presenti in env o file
        self._load_existing()

    def _env_file(self) -> Path:
        return Path.home() / ".config" / "meeting_copilot" / ".env"

    def _load_existing(self) -> None:
        """Pre-fill da env vars o file .env esistente."""
        import os
        env_file = self._env_file()
        existing: dict[str, str] = {}
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip().strip('"').strip("'")
        ak = existing.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
        dk = existing.get("DEEPGRAM_API_KEY") or os.environ.get("DEEPGRAM_API_KEY", "")
        su = existing.get("MEETING_COPILOT_SERVER_URL") or os.environ.get("MEETING_COPILOT_SERVER_URL", "")
        sk = existing.get("MEETING_COPILOT_API_KEY") or os.environ.get("MEETING_COPILOT_API_KEY", "")
        if ak: self.input_anth.setText(ak)
        if dk: self.input_dg.setText(dk)
        if su: self.input_srv_url.setText(su)
        if sk: self.input_srv_key.setText(sk)

    def save(self) -> bool:
        """Salva su file .env. Ritorna True se almeno Anthropic key valida."""
        ak = self.input_anth.text().strip()
        dk = self.input_dg.text().strip()
        su = self.input_srv_url.text().strip()
        sk = self.input_srv_key.text().strip()
        if not ak.startswith("sk-ant-"):
            self.lbl_status.setText("⚠️ Anthropic key sembra non valida (deve iniziare con sk-ant-).")
            self.lbl_status.setStyleSheet(f"color:{THEME.palette.warning}; font-size:12px;")
            return False
        env_file = self._env_file()
        env_file.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f'ANTHROPIC_API_KEY={ak}',
            f'DEEPGRAM_API_KEY={dk}' if dk else '# DEEPGRAM_API_KEY=',
            f'MEETING_COPILOT_SERVER_URL={su}' if su else '# MEETING_COPILOT_SERVER_URL=',
            f'MEETING_COPILOT_API_KEY={sk}' if sk else '# MEETING_COPILOT_API_KEY=',
        ]
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        env_file.chmod(0o600)  # solo owner readable (sicurezza)
        # Imposta nelle env var del processo corrente
        import os
        os.environ["ANTHROPIC_API_KEY"] = ak
        if dk: os.environ["DEEPGRAM_API_KEY"] = dk
        if su: os.environ["MEETING_COPILOT_SERVER_URL"] = su
        if sk: os.environ["MEETING_COPILOT_API_KEY"] = sk
        self.lbl_status.setText("✓ Salvate in ~/.config/meeting_copilot/.env")
        self.lbl_status.setStyleSheet(f"color:{THEME.palette.success}; font-size:12px;")
        return True


class _StepAudio(QWidget):
    def __init__(self):
        super().__init__()
        from PyQt6.QtCore import QTimer
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 30, 40, 30)
        lay.setSpacing(12)

        t = QLabel("Setup audio")
        t.setStyleSheet(qss.title_h1())
        lay.addWidget(t)

        sub = QLabel(
            "Devi creare 2 device audio in Audio MIDI Setup. È un setup di 1 min "
            "che però va fatto a mano (Apple non permette di crearli da codice)."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet(qss.body_dim())
        lay.addWidget(sub)

        # Status check labels (popolate dal check, auto-refresh)
        self.lbl_blackhole = QLabel()
        self.lbl_aggregate = QLabel()
        self.lbl_multiout = QLabel()
        for w in (self.lbl_blackhole, self.lbl_aggregate, self.lbl_multiout):
            w.setStyleSheet(qss.body() + "padding:3px 0; font-size:14px;")
            lay.addWidget(w)

        # Button row: apri Audio MIDI Setup
        btn_row = QHBoxLayout()
        self.btn_open_ams = QPushButton("🎙️  Apri Audio MIDI Setup")
        self.btn_open_ams.setObjectName("primary")
        self.btn_open_ams.setStyleSheet(qss.primary_button())
        self.btn_open_ams.clicked.connect(self._open_audio_midi_setup)
        btn_row.addWidget(self.btn_open_ams)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        # Istruzioni step-by-step
        instr = QLabel(
            "<div style='margin-top:14px; padding:12px; background:rgba(139,92,246,0.08); "
            "border-radius:8px; border-left:3px solid #8B5CF6;'>"
            "<p style='margin:0 0 8px 0; font-weight:600; color:#8B5CF6;'>Cosa fare nella finestra:</p>"
            "<p style='margin:6px 0; font-size:13px; line-height:1.55;'>"
            "<b>1.</b> Click <b>+</b> in basso a sinistra → <i>Crea dispositivo aggregato</i><br>"
            "<b>2.</b> Spunta <b>BlackHole 2ch</b> + <b>il tuo microfono</b><br>"
            "<b>3.</b> Doppio-click sul nome del nuovo device → rinomina <b>MeetingCapture</b><br>"
            "<br>"
            "<b>4.</b> Click <b>+</b> ancora → <i>Crea dispositivo multi-output</i><br>"
            "<b>5.</b> Spunta <b>BlackHole 2ch</b> + <b>i tuoi altoparlanti</b><br>"
            "<b>6.</b> Rinomina <b>MeetingMonitor</b>"
            "</p>"
            "</div>"
        )
        instr.setWordWrap(True)
        instr.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(instr)

        # Auto-refresh status ogni 2 secondi mentre user è in Audio MIDI Setup
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(2000)
        self._poll_timer.timeout.connect(self.refresh_status)

        self.refresh_status()

    def showEvent(self, event):
        super().showEvent(event)
        # Start polling when this step is shown
        self._poll_timer.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._poll_timer.stop()

    def _open_audio_midi_setup(self) -> None:
        """Apre Audio MIDI Setup.app via osascript (AppleScript)."""
        import subprocess
        from pathlib import Path
        # Cerca il file AppleScript (bundled vs dev mode)
        try:
            from ..paths import bundle_resources_dir
            resources = bundle_resources_dir()
        except Exception:
            resources = None
        # Path candidates: bundle resources, then repo deploy/
        candidates = []
        if resources:
            candidates.append(resources / "deploy" / "open_audio_midi_setup.applescript")
        candidates.append(Path(__file__).resolve().parents[3] / "deploy" / "open_audio_midi_setup.applescript")
        applescript = next((p for p in candidates if p.exists()), None)
        try:
            if applescript:
                subprocess.run(["osascript", str(applescript)], check=False, timeout=5)
            else:
                # Fallback: open generico
                subprocess.run(["open", "-a", "Audio MIDI Setup"], check=False, timeout=5)
        except Exception as e:
            print(f"[onboarding] errore apertura AMS: {e}")

    def refresh_status(self) -> None:
        """Verifica blackhole/aggregate/multioutput presenti in sounddevice."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            names = " | ".join(d["name"].lower() for d in devices)
        except Exception as e:
            self.lbl_blackhole.setText(f"⚠️  Impossibile leggere device audio: {e}")
            return

        def chk(label: str, needle: str, helper: str) -> str:
            if needle in names:
                return f"<span style='color:{THEME.palette.success};'>✓</span>  {label} trovato"
            return (
                f"<span style='color:{THEME.palette.warning};'>○</span>  {label} mancante "
                f"<span style='color:{THEME.palette.text_tertiary}; font-size:11px;'>"
                f"({helper})</span>"
            )

        self.lbl_blackhole.setText(chk("BlackHole 2ch", "blackhole", "brew install blackhole-2ch"))
        self.lbl_aggregate.setText(chk("Aggregate Device 'MeetingCapture'", "meetingcapture",
                                        "Audio MIDI Setup → Aggregate Device"))
        self.lbl_multiout.setText(chk("Multi-Output 'MeetingMonitor'", "meetingmonitor",
                                       "Audio MIDI Setup → Multi-Output Device"))

    def _open_setup_guide(self):
        import subprocess
        # Apre il file SETUP.md del progetto
        try:
            from pathlib import Path
            setup_md = Path(__file__).resolve().parents[3] / "SETUP.md"
            if setup_md.exists():
                subprocess.run(["open", str(setup_md)])
        except Exception:
            pass


class _StepKB(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 40, 40, 40)
        lay.setSpacing(16)
        lay.addStretch()

        t = QLabel("La tua prima knowledge section")
        t.setStyleSheet(qss.title_h1())
        lay.addWidget(t)

        sub = QLabel(
            "Una 'sezione' è un gruppo di documenti che usi in un certo tipo di call. "
            "Es: 'Pitch ProductB', 'Cliente Pharma', 'Obiezioni tipiche'.\n"
            "Crea ora la tua prima sezione — potrai aggiungere altri documenti dopo."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet(qss.body_dim())
        lay.addWidget(sub)

        self.input_name = QLineEdit()
        self.input_name.setPlaceholderText("Nome sezione (es. Pitch ProductB)")
        self.input_name.setStyleSheet(qss.line_edit())
        lay.addWidget(self.input_name)

        self.lbl_result = QLabel("")
        self.lbl_result.setStyleSheet(qss.body_dim())
        lay.addWidget(self.lbl_result)

        btn_create = QPushButton("Crea sezione")
        btn_create.setObjectName("primary")
        btn_create.setStyleSheet(qss.primary_button())
        btn_create.clicked.connect(self._on_create)
        lay.addWidget(btn_create, alignment=Qt.AlignmentFlag.AlignLeft)

        skip = QLabel("Puoi saltare e crearla dopo dal Knowledge Manager.")
        skip.setStyleSheet(f"color:{THEME.palette.text_tertiary}; font-size:11px;")
        lay.addWidget(skip)

        lay.addStretch()

    def _on_create(self):
        name = self.input_name.text().strip()
        if not name:
            self.lbl_result.setText("⚠️ Inserisci un nome.")
            return
        try:
            section = KnowledgeManager.create_section(name)
            # Attivala subito
            active = KnowledgeManager.get_active_sections()
            if section.slug not in active:
                active.append(section.slug)
                KnowledgeManager.set_active_sections(active)
            self.lbl_result.setText(
                f"✓ Sezione '{section.slug}' creata e attivata. "
                f"Aggiungi documenti dopo dal Knowledge Manager."
            )
            self.lbl_result.setStyleSheet(f"color:{THEME.palette.success}; font-size:13px;")
        except Exception as e:
            self.lbl_result.setText(f"Errore: {e}")
            self.lbl_result.setStyleSheet(f"color:{THEME.palette.danger}; font-size:13px;")


class _StepHotkeys(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 40, 40, 40)
        lay.setSpacing(16)
        lay.addStretch()

        t = QLabel("Hotkey & scorciatoie")
        t.setStyleSheet(qss.title_h1())
        lay.addWidget(t)

        sub = QLabel("Le hotkey funzionano anche quando l'overlay non è in primo piano.")
        sub.setWordWrap(True)
        sub.setStyleSheet(qss.body_dim())
        lay.addWidget(sub)

        # Cheat sheet
        rows = [
            ("Cmd+Shift+R", "💡 Ask — triggera Analyzer → Composer ora"),
            ("Cmd+Shift+S", "Salva ultimo passaggio in una sezione KB"),
            ("Cmd+Shift+N", "Focus pannello note"),
            ("Cmd+Shift+T", "Mostra/nascondi overlay"),
            ("Esc",         "Chiudi overlay"),
        ]
        sheet = "<table style='margin-top:12px;'>"
        for k, desc in rows:
            sheet += (
                f"<tr>"
                f"<td style='padding:6px 12px 6px 0; font-family:Menlo; "
                f"color:{THEME.palette.accent_primary}; font-size:13px; "
                f"vertical-align:top;'><b>{k}</b></td>"
                f"<td style='padding:6px 0; color:{THEME.palette.text_primary}; "
                f"font-size:13px;'>{desc}</td>"
                f"</tr>"
            )
        sheet += "</table>"
        lbl = QLabel(sheet)
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(lbl)

        note = QLabel(
            "Per attivare le hotkey global, macOS chiederà permesso Accessibility. "
            "Vai in System Settings → Privacy & Security → Accessibility e attiva il terminale "
            "(o l'app) da cui lanci Meeting Copilot."
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            f"color:{THEME.palette.text_tertiary}; font-size:12px; "
            f"line-height:1.6; margin-top:16px;"
        )
        lay.addWidget(note)

        lay.addStretch()


class _StepDone(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 40, 40, 40)
        lay.setSpacing(16)
        lay.addStretch()

        t = QLabel("Sei pronto")
        t.setStyleSheet(qss.title_h1())
        lay.addWidget(t)

        sub = QLabel(
            "Premi 'Inizia' qui sotto. L'overlay apparirà sopra le altre app, "
            "invisibile durante screen share.\n\n"
            "Quando parte una call, il Watcher Agent osserverà la conversazione e "
            "ti mostrerà 'ganci' cliccabili quando rileva domande o opportunità rilevanti."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet(qss.body_dim() + "line-height:1.7;")
        lay.addWidget(sub)

        lay.addStretch()


# ─────────────────────────────────────────────────────────────
# Wizard container
# ─────────────────────────────────────────────────────────────
class OnboardingWizard(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Meeting Copilot — Onboarding")
        self.resize(640, 540)
        self.setModal(True)
        self.setStyleSheet(
            f"QDialog {{ background-color: {THEME.palette.bg_elevated}; }}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Stacked pages
        self.stack = QStackedWidget()
        self.pages = [
            _StepWelcome(),
            _StepApiKeys(),     # NEW: API keys configuration (step 2)
            _StepAudio(),
            _StepKB(),
            _StepHotkeys(),
            _StepDone(),
        ]
        for p in self.pages:
            self.stack.addWidget(p)
        root.addWidget(self.stack, 1)

        # Footer bar: indicator + buttons
        footer = QHBoxLayout()
        footer.setContentsMargins(40, 16, 40, 24)

        self.lbl_step = QLabel("1 / 5")
        self.lbl_step.setStyleSheet(qss.body_dim())
        footer.addWidget(self.lbl_step)
        footer.addStretch()

        self.btn_back = QPushButton("Indietro")
        self.btn_back.setProperty("class", "secondary")
        self.btn_back.setStyleSheet(qss.secondary_button())
        self.btn_back.clicked.connect(self._on_back)
        footer.addWidget(self.btn_back)

        self.btn_next = QPushButton("Avanti  →")
        self.btn_next.setObjectName("primary")
        self.btn_next.setStyleSheet(qss.primary_button())
        self.btn_next.clicked.connect(self._on_next)
        footer.addWidget(self.btn_next)
        root.addLayout(footer)

        self._update_buttons()

    # ── navigation ──
    def _on_back(self):
        i = self.stack.currentIndex()
        if i > 0:
            self.stack.setCurrentIndex(i - 1)
        self._update_buttons()

    def _on_next(self):
        i = self.stack.currentIndex()
        # Step 1 = API keys: salva prima di andare avanti
        if i == 1:
            if not self.pages[1].save():
                return  # blocca se Anthropic key non valida
        # Step 2 = Audio: refresh status
        if i == 2:
            self.pages[2].refresh_status()
        if i < len(self.pages) - 1:
            self.stack.setCurrentIndex(i + 1)
            self._update_buttons()
        else:
            mark_onboarded()
            self.accept()

    def _update_buttons(self):
        i = self.stack.currentIndex()
        n = len(self.pages)
        self.lbl_step.setText(f"{i + 1} / {n}")
        self.btn_back.setEnabled(i > 0)
        self.btn_next.setText("Inizia  ✓" if i == n - 1 else "Avanti  →")


# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────
def run_onboarding_if_needed(force: bool = False) -> None:
    """Mostra l'onboarding modale se necessario."""
    if not force and is_onboarded():
        return
    app = QApplication.instance() or QApplication(sys.argv)
    w = OnboardingWizard()
    w.exec()


if __name__ == "__main__":
    run_onboarding_if_needed(force=True)
