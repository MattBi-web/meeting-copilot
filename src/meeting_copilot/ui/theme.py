"""Design tokens — palette + tipografia + spacing.

Tutti i moduli UI (overlay, onboarding, settings) leggono da qui.
Cambi globali di design = cambi un solo file.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    # Background scura, tonalità tendente al blu (più moderna del puro nero)
    bg_base: str = "rgba(10, 11, 15, 240)"          # finestra principale
    bg_surface: str = "rgba(20, 22, 28, 235)"       # pannelli interni
    bg_elevated: str = "rgba(28, 30, 38, 240)"      # popup, dialog
    bg_hover: str = "rgba(255, 255, 255, 12)"
    bg_active: str = "rgba(255, 255, 255, 20)"

    # Borders sottili, "vetro"
    border_subtle: str = "rgba(255, 255, 255, 18)"
    border_strong: str = "rgba(255, 255, 255, 30)"
    border_focus: str = "rgba(59, 130, 246, 120)"

    # Text gerarchy
    text_primary: str = "#F3F4F6"      # 96% bianco, gradevole
    text_secondary: str = "#9CA3AF"    # grigio caldo
    text_tertiary: str = "#6B7280"     # quasi solo per label
    text_disabled: str = "#4B5563"

    # Accent (semantico)
    accent_primary: str = "#3B82F6"    # blu — primario, link, focus
    accent_primary_dim: str = "rgba(59, 130, 246, 30)"
    success: str = "#10B981"           # verde — done, ready, thumbs_up
    warning: str = "#F59E0B"           # arancione — analyzing, hook
    danger: str = "#EF4444"            # rosso — errori, thumbs_down
    info: str = "#60A5FA"              # azzurro chiaro — informativo


@dataclass(frozen=True)
class Spacing:
    xs: int = 4
    sm: int = 8
    md: int = 12
    lg: int = 16
    xl: int = 24
    xxl: int = 32


@dataclass(frozen=True)
class Typography:
    # Famiglia: usa il system font macOS (NO "SF Pro Text" perché Qt non lo trova)
    family: str = "-apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif"
    family_mono: str = "Menlo, 'SF Mono', monospace"

    # Sizes (px)
    label: int = 10        # uppercase section labels
    caption: int = 11      # metadata, timestamps
    body: int = 13         # default
    body_lg: int = 14
    title: int = 16
    title_lg: int = 22

    # Letter spacing per le labels uppercase
    label_letter_spacing: str = "1.2px"


@dataclass(frozen=True)
class Radius:
    sm: int = 6
    md: int = 8
    lg: int = 12
    xl: int = 16


@dataclass(frozen=True)
class Theme:
    palette: Palette = Palette()
    spacing: Spacing = Spacing()
    typography: Typography = Typography()
    radius: Radius = Radius()


THEME = Theme()


# ─────────────────────────────────────────────────────────────
# QSS templates — chiamare qss.X() per uno stylesheet pronto
# ─────────────────────────────────────────────────────────────
class _QSS:
    """Templates Qt StyleSheet pronti per i widget più usati."""

    @staticmethod
    def container_root() -> str:
        """Container principale dell'overlay."""
        return f"""
        QFrame#container {{
            background-color: {THEME.palette.bg_base};
            border-radius: {THEME.radius.lg}px;
            border: 1px solid {THEME.palette.border_subtle};
        }}
        """

    @staticmethod
    def section_label() -> str:
        """Label tipo 'TRASCRIZIONE LIVE' uppercase."""
        return (
            f"color: {THEME.palette.text_tertiary}; "
            f"font-size: {THEME.typography.label}px; "
            f"font-weight: 600; "
            f"text-transform: uppercase; "
            f"letter-spacing: {THEME.typography.label_letter_spacing}; "
            f"padding: {THEME.spacing.sm}px {THEME.spacing.md}px {THEME.spacing.xs}px {THEME.spacing.md}px;"
        )

    @staticmethod
    def text_view() -> str:
        """QTextEdit content area (trascrizione, suggerimenti, note)."""
        return f"""
        QTextEdit {{
            background-color: transparent;
            color: {THEME.palette.text_primary};
            border: none;
            padding: {THEME.spacing.xs}px {THEME.spacing.md}px {THEME.spacing.md}px {THEME.spacing.md}px;
            font-size: {THEME.typography.body}px;
            line-height: 1.5;
            selection-background-color: {THEME.palette.accent_primary_dim};
        }}
        QTextEdit:focus {{
            border: none;   /* niente focus ring rumoroso */
        }}
        """

    @staticmethod
    def header_button() -> str:
        """Bottone piccolo nell'header dell'overlay."""
        return f"""
        QPushButton {{
            background-color: transparent;
            color: {THEME.palette.text_secondary};
            border: none;
            font-size: {THEME.typography.caption}px;
            font-weight: 500;
            padding: {THEME.spacing.xs}px {THEME.spacing.sm}px;
            border-radius: {THEME.radius.sm}px;
        }}
        QPushButton:hover {{
            background-color: {THEME.palette.bg_hover};
            color: {THEME.palette.text_primary};
        }}
        QPushButton:pressed {{
            background-color: {THEME.palette.bg_active};
        }}
        QPushButton:checked {{
            background-color: {THEME.palette.accent_primary_dim};
            color: {THEME.palette.accent_primary};
        }}
        """

    @staticmethod
    def primary_button() -> str:
        """Bottone CTA primario (es. 'Avanti' nel wizard)."""
        return f"""
        QPushButton#primary {{
            background-color: {THEME.palette.accent_primary};
            color: white;
            border: none;
            padding: {THEME.spacing.sm}px {THEME.spacing.lg}px;
            border-radius: {THEME.radius.md}px;
            font-size: {THEME.typography.body}px;
            font-weight: 600;
        }}
        QPushButton#primary:hover {{
            background-color: #2563EB;
        }}
        QPushButton#primary:disabled {{
            background-color: {THEME.palette.text_disabled};
            color: {THEME.palette.text_tertiary};
        }}
        """

    @staticmethod
    def secondary_button() -> str:
        return f"""
        QPushButton.secondary {{
            background-color: {THEME.palette.bg_elevated};
            color: {THEME.palette.text_primary};
            border: 1px solid {THEME.palette.border_subtle};
            padding: {THEME.spacing.sm}px {THEME.spacing.lg}px;
            border-radius: {THEME.radius.md}px;
            font-size: {THEME.typography.body}px;
            font-weight: 500;
        }}
        QPushButton.secondary:hover {{
            border-color: {THEME.palette.border_strong};
        }}
        """

    @staticmethod
    def line_edit() -> str:
        return f"""
        QLineEdit {{
            background-color: {THEME.palette.bg_surface};
            color: {THEME.palette.text_primary};
            border: 1px solid {THEME.palette.border_subtle};
            border-radius: {THEME.radius.md}px;
            padding: {THEME.spacing.sm}px {THEME.spacing.md}px;
            font-size: {THEME.typography.body}px;
        }}
        QLineEdit:focus {{
            border-color: {THEME.palette.border_focus};
        }}
        """

    @staticmethod
    def hooks_list() -> str:
        return f"""
        QListWidget {{
            background-color: transparent;
            border: none;
            padding: {THEME.spacing.xs}px {THEME.spacing.sm}px;
        }}
        QListWidget::item {{
            color: {THEME.palette.info};
            padding: {THEME.spacing.sm}px {THEME.spacing.md}px;
            border-radius: {THEME.radius.sm}px;
            border: 1px solid {THEME.palette.accent_primary_dim};
            margin: 2px 0;
        }}
        QListWidget::item:hover {{
            background-color: {THEME.palette.accent_primary_dim};
            color: {THEME.palette.text_primary};
        }}
        QListWidget::item:selected {{
            background-color: rgba(59,130,246,80);
            color: {THEME.palette.text_primary};
        }}
        """

    @staticmethod
    def title_h1() -> str:
        return (
            f"color: {THEME.palette.text_primary}; "
            f"font-size: {THEME.typography.title_lg}px; "
            f"font-weight: 700;"
        )

    @staticmethod
    def title_h2() -> str:
        return (
            f"color: {THEME.palette.text_primary}; "
            f"font-size: {THEME.typography.title}px; "
            f"font-weight: 600;"
        )

    @staticmethod
    def body() -> str:
        return (
            f"color: {THEME.palette.text_primary}; "
            f"font-size: {THEME.typography.body}px; "
            f"line-height: 1.5;"
        )

    @staticmethod
    def body_dim() -> str:
        return (
            f"color: {THEME.palette.text_secondary}; "
            f"font-size: {THEME.typography.body}px;"
        )


qss = _QSS()
