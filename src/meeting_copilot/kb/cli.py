"""CLI per gestire la knowledge base da terminale.

Comandi disponibili:
    meeting-copilot kb list
    meeting-copilot kb create "Nome Sezione"
    meeting-copilot kb ingest <slug> file1.pdf file2.docx
    meeting-copilot kb reindex <slug>
    meeting-copilot kb reindex-all
    meeting-copilot kb search <slug> "query naturale"
    meeting-copilot kb delete <slug>
    meeting-copilot kb activate <slug>
    meeting-copilot kb deactivate <slug>
    meeting-copilot kb status
"""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .manager import KnowledgeManager

app = typer.Typer(help="Knowledge base management commands.")
console = Console()


@app.command("list")
def list_sections():
    """Elenca tutte le sezioni esistenti."""
    sections = KnowledgeManager.list_sections()
    if not sections:
        console.print("[dim]Nessuna sezione. Crea con: meeting-copilot kb create 'Nome'[/]")
        return

    active = set(KnowledgeManager.get_active_sections())
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("●")
    table.add_column("Slug")
    table.add_column("Sources", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Last reindex")
    for s in sections:
        active_mark = "[green]●[/]" if s.slug in active else " "
        table.add_row(
            active_mark,
            s.slug,
            str(s.n_sources),
            str(s.n_chunks),
            s.last_reindex or "[dim]never[/]",
        )
    console.print(table)
    console.print(f"\n[dim]Attive: {', '.join(sorted(active)) or '(nessuna)'}[/]")


@app.command("create")
def create_section(name: str = typer.Argument(..., help="Nome umano della sezione, es. 'Pitch ProductB'")):
    """Crea una nuova sezione (cartella + _urls.txt placeholder)."""
    section = KnowledgeManager.create_section(name)
    console.print(f"[green]✓[/] Sezione creata: [bold]{section.slug}[/]")
    console.print(f"  Cartella: {section.sources_dir}")
    console.print("  Aggiungi file con: [cyan]meeting-copilot kb ingest "
                  f"{section.slug} file1.pdf file2.docx[/]")
    console.print("  Oppure copia file dentro la cartella e fai: "
                  f"[cyan]meeting-copilot kb reindex {section.slug}[/]")


@app.command("ingest")
def ingest(
    slug: str = typer.Argument(..., help="Slug della sezione"),
    files: list[Path] = typer.Argument(..., help="File da aggiungere"),
):
    """Copia file in sources/ e fa reindex."""
    copied = KnowledgeManager.add_files(slug, [Path(f) for f in files])
    if not copied:
        console.print("[yellow]Nessun file copiato (file non esistenti?)[/]")
        return
    for p in copied:
        console.print(f"  + {p.name}")
    console.print(f"[green]✓[/] {len(copied)} file copiati. Reindexing…")
    KnowledgeManager.reindex(slug, verbose=True)


@app.command("reindex")
def reindex(slug: str):
    """Re-indicizza una sezione. Riprocessa TUTTI i file in sources/."""
    KnowledgeManager.reindex(slug, verbose=True)


@app.command("reindex-all")
def reindex_all():
    """Re-indicizza tutte le sezioni."""
    sections = KnowledgeManager.list_sections()
    for s in sections:
        console.print(f"\n[bold]→ {s.slug}[/]")
        KnowledgeManager.reindex(s.slug, verbose=True)


@app.command("search")
def search_cmd(
    slug: str = typer.Argument(..., help="Slug sezione (oppure 'active' per le attive)"),
    query: str = typer.Argument(..., help="Query naturale"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Numero di hits"),
):
    """Test query manuale su una sezione."""
    sections = KnowledgeManager.get_active_sections() if slug == "active" else [slug]
    hits = KnowledgeManager.search(query, sections=sections, top_k=top_k)
    if not hits:
        console.print("[dim]Nessun hit.[/]")
        return
    for i, h in enumerate(hits, 1):
        console.print(f"\n[bold cyan]#{i}[/] [dim]score={h.score:.3f}[/] [yellow]{h.section}[/] · "
                      f"[magenta]{h.source}[/]")
        console.print(h.text[:400] + ("…" if len(h.text) > 400 else ""))


@app.command("delete")
def delete_section(
    slug: str,
    yes: bool = typer.Option(False, "--yes", "-y", help="Conferma senza prompt"),
):
    """Elimina una sezione (sources + index)."""
    if not yes:
        confirm = typer.confirm(f"Eliminare DEFINITIVAMENTE la sezione '{slug}' (sources + index)?")
        if not confirm:
            console.print("Annullato.")
            return
    KnowledgeManager.delete_section(slug)
    console.print(f"[red]✗[/] Sezione '{slug}' eliminata.")


@app.command("activate")
def activate(slug: str):
    """Attiva una sezione per il retrieval runtime."""
    active = KnowledgeManager.get_active_sections()
    if slug not in active:
        active.append(slug)
        KnowledgeManager.set_active_sections(active)
    console.print(f"[green]✓[/] '{slug}' attivata.")


@app.command("deactivate")
def deactivate(slug: str):
    """Disattiva una sezione."""
    active = KnowledgeManager.get_active_sections()
    if slug in active:
        active.remove(slug)
        KnowledgeManager.set_active_sections(active)
    console.print(f"[yellow]○[/] '{slug}' disattivata.")


@app.command("status")
def status():
    """Mostra status attuale (sezioni attive)."""
    active = KnowledgeManager.get_active_sections()
    if not active:
        console.print("[dim]Nessuna sezione attiva.[/]")
        return
    console.print("[bold]Sezioni attive:[/]")
    for slug in active:
        console.print(f"  [green]●[/] {slug}")


if __name__ == "__main__":
    app()
