"""CLI graph — rebuild, stats, search (debug).

Comandi:
    meeting-copilot graph rebuild <slug>
    meeting-copilot graph stats <slug>
    meeting-copilot graph entities <slug>     # top N per mention_count
    meeting-copilot graph search <slug> "query"   # debug graph retrieval
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from ...config import GRAPH
from .builder import build_graph
from .retriever import graph_search
from .store import GraphStore

app = typer.Typer(help="Knowledge Graph (Fase 17) management.")
console = Console()


@app.command("rebuild")
def rebuild(
    slug: str = typer.Argument(..., help="Slug della sezione KB"),
    max_workers: int = typer.Option(3, "--workers", "-w",
                                    help="Thread parallel per LLM extraction"),
):
    """Estrae entità/relazioni da tutti i chunks della sezione e ricostruisce il grafo."""
    if not GRAPH.enabled:
        console.print(
            "[yellow]GRAPH.enabled=False[/]. Attiva in config.py per usare il grafo."
        )
        console.print("(Il rebuild procede comunque per scopo di test.)")
        # Forziamo abilitazione locale per il rebuild manuale
        import meeting_copilot.config as _cfg
        from dataclasses import replace
        _cfg.GRAPH = replace(_cfg.GRAPH, enabled=True)

    stats = build_graph(slug, verbose=True, max_workers=max_workers)
    console.print(f"\n[green]✓[/] Grafo costruito:")
    for k, v in stats.items():
        console.print(f"   {k}: {v}")


@app.command("stats")
def stats(slug: str = typer.Argument(..., help="Slug della sezione KB")):
    """Mostra statistiche del grafo per la sezione."""
    s = GraphStore(slug).stats()
    t = Table(show_header=True, header_style="bold cyan",
              title=f"Graph stats — '{slug}'")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    for k, v in s.items():
        t.add_row(k, str(v))
    console.print(t)


@app.command("entities")
def list_entities(
    slug: str = typer.Argument(..., help="Slug della sezione KB"),
    top: int = typer.Option(20, "--top", "-n"),
):
    """Top N entità per mention_count."""
    entities = GraphStore(slug).list_entities()[:top]
    if not entities:
        console.print(f"[dim]Nessuna entità nel grafo di '{slug}'.[/]")
        return
    t = Table(show_header=True, header_style="bold cyan",
              title=f"Top entities — '{slug}'")
    t.add_column("ID", justify="right")
    t.add_column("Name")
    t.add_column("Type")
    t.add_column("Mentions", justify="right")
    t.add_column("Has embed", justify="center")
    for e in entities:
        t.add_row(
            str(e.id), e.name, e.type, str(e.mention_count),
            "✓" if e.embedding else "—",
        )
    console.print(t)


@app.command("search")
def search(
    slug: str = typer.Argument(..., help="Slug della sezione KB"),
    query: str = typer.Argument(..., help="Query naturale"),
    top: int = typer.Option(5, "--top", "-k"),
):
    """Debug: esegui graph retrieval e mostra i chunks recuperati."""
    if not GRAPH.enabled:
        console.print(
            "[yellow]Nota: GRAPH.enabled=False. Forzo enabled per test...[/]"
        )
        import meeting_copilot.config as _cfg
        from dataclasses import replace
        _cfg.GRAPH = replace(_cfg.GRAPH, enabled=True)

    hits = graph_search(query, [slug], top_k=top)
    if not hits:
        console.print("[dim]Nessun chunk recuperato dal grafo.[/]")
        return
    for i, h in enumerate(hits, 1):
        console.print(f"\n[cyan]#{i}[/] score=[yellow]{h.score:.2f}[/] "
                      f"src=[magenta]{h.source}[/]")
        console.print(h.text[:300] + ("…" if len(h.text) > 300 else ""))


if __name__ == "__main__":
    app()
