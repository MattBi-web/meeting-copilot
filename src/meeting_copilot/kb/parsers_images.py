"""Image extraction da PPTX — Fase 20.1.

Per ogni .pptx nella sezione, rende OGNI slide come PNG (1280x720) e salva nella
sottocartella `images/` della sezione. Mantiene metadata: source_file, slide_idx,
ratio. Le immagini vengono poi embeddate (Fase 20.2) e indicizzate (Fase 20.3).

Implementazione: pptx-renderer via LibreOffice + pdf2image, OPPURE python-pptx
con rendering shape-by-shape (no — qualità scarsa). Per produrre slide-as-image
in modo affidabile, usiamo soffice (LibreOffice headless) → PDF → pdf2image → PNG.
Se LibreOffice non è installato, fallback: estrai SOLO le immagini embed nel pptx
(senza testo/layout rendered).

Nessuna dipendenza nuova obbligatoria — il modulo si degrada se manca soffice o
pdf2image. La feature è opt-in via IMAGE.enabled.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import paths


@dataclass(frozen=True)
class SlideImage:
    """Una slide renderizzata come PNG su disco."""
    slide_idx: int             # 1-based
    source_file: str           # nome del .pptx originale
    image_path: Path           # path assoluto del PNG salvato
    width: int
    height: int


def images_dir(slug: str) -> Path:
    """Sottocartella della sezione dove vengono salvati i PNG."""
    d = paths.section_dir(slug) / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ──────────────────────────────────────────────────────────────
# Strategy 1: LibreOffice headless → PDF → PNG (preferred, quality top)
# ──────────────────────────────────────────────────────────────
def _has_libreoffice() -> bool:
    return shutil.which("soffice") is not None or shutil.which("libreoffice") is not None


def _libreoffice_bin() -> str:
    return shutil.which("soffice") or shutil.which("libreoffice") or "soffice"


def _has_pdf2image() -> bool:
    try:
        import pdf2image  # noqa: F401
        return True
    except ImportError:
        return False


def _render_pptx_to_png_via_lo(
    pptx_path: Path, out_dir: Path, dpi: int = 150,
) -> list[SlideImage]:
    """Converti .pptx → PDF (via LibreOffice) → PNG (via pdf2image)."""
    if not _has_libreoffice():
        raise RuntimeError("LibreOffice (soffice) non trovato in PATH")
    if not _has_pdf2image():
        raise RuntimeError("pdf2image non installato. Installa: uv pip install pdf2image")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        # Step 1: PPTX → PDF
        subprocess.run(
            [_libreoffice_bin(), "--headless", "--convert-to", "pdf",
             "--outdir", str(tmp_dir), str(pptx_path)],
            check=True, capture_output=True, timeout=180,
        )
        pdf_files = list(tmp_dir.glob("*.pdf"))
        if not pdf_files:
            raise RuntimeError(f"LibreOffice non ha prodotto un PDF per {pptx_path.name}")
        pdf_path = pdf_files[0]

        # Step 2: PDF → list[PIL.Image]
        from pdf2image import convert_from_path
        pages = convert_from_path(str(pdf_path), dpi=dpi)

        # Step 3: salva PNG con nome stabile
        results: list[SlideImage] = []
        stem = pptx_path.stem
        for i, page in enumerate(pages, start=1):
            out_path = out_dir / f"{stem}__slide_{i:03d}.png"
            page.save(out_path, "PNG", optimize=True)
            results.append(SlideImage(
                slide_idx=i, source_file=pptx_path.name,
                image_path=out_path, width=page.width, height=page.height,
            ))
        return results


# ──────────────────────────────────────────────────────────────
# Strategy 2: estrai SOLO immagini embedded (fallback)
# ──────────────────────────────────────────────────────────────
def _extract_embedded_images(
    pptx_path: Path, out_dir: Path,
) -> list[SlideImage]:
    """Fallback: estrai le immagini embed (PNG/JPEG) dalle slide via python-pptx.

    NON è una vera "slide renderizzata" — è solo le immagini interne
    (loghi, screenshot inseriti). Utile se LibreOffice manca.
    """
    try:
        from pptx import Presentation
    except ImportError:
        raise RuntimeError("python-pptx non installato")

    prs = Presentation(str(pptx_path))
    results: list[SlideImage] = []
    stem = pptx_path.stem

    for slide_idx, slide in enumerate(prs.slides, start=1):
        for shape_idx, shape in enumerate(slide.shapes):
            if not getattr(shape, "image", None):
                continue
            try:
                blob = shape.image.blob
                ext = (shape.image.ext or "png").lower()
                out_path = out_dir / f"{stem}__slide_{slide_idx:03d}_shape_{shape_idx}.{ext}"
                out_path.write_bytes(blob)
                # Width/height dal shape (EMU → px conversion rough)
                w = int(shape.width / 9525) if shape.width else 0
                h = int(shape.height / 9525) if shape.height else 0
                results.append(SlideImage(
                    slide_idx=slide_idx, source_file=pptx_path.name,
                    image_path=out_path, width=w, height=h,
                ))
            except Exception as e:
                print(f"[parsers_images] embed extract skipped: {e}")
                continue
    return results


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────
def extract_pptx_images(
    pptx_path: Path,
    slug: str,
    *,
    prefer_render: bool = True,
) -> list[SlideImage]:
    """Estrae le slide del .pptx come immagini PNG nella sezione `slug`.

    Args:
        pptx_path: path al file .pptx
        slug: sezione KB di destinazione
        prefer_render: se True, prova LibreOffice rendering; fallback a embedded.

    Returns:
        Lista di SlideImage. Vuota se errore o file non supportato.
    """
    if not pptx_path.exists() or pptx_path.suffix.lower() != ".pptx":
        return []

    out_dir = images_dir(slug)

    if prefer_render and _has_libreoffice() and _has_pdf2image():
        try:
            return _render_pptx_to_png_via_lo(pptx_path, out_dir)
        except Exception as e:
            print(f"[parsers_images] LibreOffice render failed ({e}), fallback to embedded")

    try:
        return _extract_embedded_images(pptx_path, out_dir)
    except Exception as e:
        print(f"[parsers_images] embedded extract failed: {e}")
        return []


def list_section_images(slug: str) -> list[Path]:
    """Tutte le immagini salvate per la sezione."""
    d = images_dir(slug)
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
