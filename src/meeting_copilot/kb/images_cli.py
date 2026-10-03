"""CLI image retrieval — rebuild / stats / search (Fase 20).

Comandi:
    meeting-copilot images rebuild <slug>
    meeting-copilot images stats <slug>
    meeting-copilot images search <slug> "query"
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from ..config import IMAGE
from .image_store import image_search, images_stats, rebuild_images

app = typer.Typer(help="Multimodal image retrieval (Fase 20) management.")
console = Console()


@app.command("rebuild")
def rebuild(slug: str = typer.Argument(..., help="Slug della sezione KB")):
    """Estrai slide da PPTX della sezione, embedda con CLIP, indicizza."""
    if not IMAGE.enabled:
        console.print(
            "[yellow]IMAGE.enabled=False[/]. Forzo locale per test."
        )
        import meeting_copilot.config as _cfg
        from dataclasses import replace
        _cfg.IMAGE = replace(_cfg.IMAGE, enabled=True)
    stats = rebuild_images(slug, verbose=True)
    console.print(f"\n[green]✓[/] Image index built:")
    for k, v in stats.items():
        console.print(f"   {k}: {v}")


@app.command("stats")
def stats(slug: str = typer.Argument(..., help="Slug della sezione KB")):
    """Mostra statistiche image index per la sezione."""
    s = images_stats(slug)
    t = Table(show_header=True, header_style="bold cyan",
              title=f"Images stats — '{slug}'")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    for k, v in s.items():
        t.add_row(k, str(v))
    console.print(t)


@app.command("search")
def search(
    slug: str = typer.Argument(..., help="Slug della sezione KB"),
    query: str = typer.Argument(..., help="Query naturale"),
    top: int = typer.Option(4, "--top", "-k"),
):
    """Trova le slide più rilevanti per la query."""
    if not IMAGE.enabled:
        import meeting_copilot.config as _cfg
        from dataclasses import replace
        _cfg.IMAGE = replace(_cfg.IMAGE, enabled=True)

    hits = image_search(query, [slug], top_k=top)
    if not hits:
        console.print("[dim]Nessuna slide rilevante.[/]")
        return
    for i, h in enumerate(hits, 1):
        console.print(f"\n[cyan]#{i}[/] score=[yellow]{h.score:.3f}[/] "
                      f"src=[magenta]{h.source_file}[/] slide=[blue]{h.slide_idx}[/]")
        console.print(f"   [dim]{h.image_path}[/]")


if __name__ == "__main__":
    app()
