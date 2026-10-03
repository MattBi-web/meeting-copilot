"""CLI memory — list / show / search / purge."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from .manager import MemoryManager

app = typer.Typer(help="Memory cross-call commands (Fase 15).")
console = Console()


@app.command("list")
def list_scopes():
    """Elenca scope e numero memorie."""
    scopes = MemoryManager.list_scopes()
    if not scopes:
        console.print("[dim]Nessuna memoria salvata ancora.[/]")
        return
    t = Table(show_header=True, header_style="bold cyan", title="Memory scopes")
    t.add_column("Scope")
    t.add_column("# memorie", justify="right")
    for scope, n in scopes:
        t.add_row(scope, str(n))
    console.print(t)
    console.print(f"\n[dim]Totale: {MemoryManager.total_count()} memorie.[/]")


@app.command("show")
def show(scope: str = typer.Argument(..., help="Scope da visualizzare")):
    """Mostra tutte le memorie di uno scope."""
    memories = MemoryManager.list_for_scope(scope)
    if not memories:
        console.print(f"[dim]Nessuna memoria nello scope '{scope}'.[/]")
        return
    t = Table(show_header=True, header_style="bold cyan",
              title=f"Memorie scope '{scope}' ({len(memories)})")
    t.add_column("Kind", width=10)
    t.add_column("Imp.", justify="right", width=6)
    t.add_column("Quando", width=14)
    t.add_column("Testo")
    from datetime import datetime
    for m in memories:
        ts = datetime.fromtimestamp(m.created_at).strftime("%Y-%m-%d %H:%M")
        emoji = "📌" if m.kind == "semantic" else "🗓"
        t.add_row(
            f"{emoji} {m.kind}",
            f"{m.importance:.2f}",
            ts,
            m.text[:120] + ("…" if len(m.text) > 120 else ""),
        )
    console.print(t)


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Query semantica"),
    scope: str = typer.Option("", "--scope", "-s", help="Scope filter (opzionale)"),
    top: int = typer.Option(5, "--top-k", "-k"),
):
    """Cerca memorie per similarity."""
    scopes = [scope] if scope else [s[0] for s in MemoryManager.list_scopes()]
    if not scopes:
        console.print("[dim]Nessuna memoria disponibile.[/]")
        return
    results = MemoryManager.search(query, scopes, top_k=top)
    if not results:
        console.print("[dim]Nessuna memoria rilevante.[/]")
        return
    for rec, score in results:
        emoji = "📌" if rec.kind == "semantic" else "🗓"
        console.print(
            f"[cyan]{score:.2f}[/] {emoji} [yellow]{rec.scope}[/] "
            f"[dim](imp {rec.importance:.2f})[/]"
        )
        console.print(f"  {rec.text}")


@app.command("delete-scope")
def delete_scope(scope: str, yes: bool = typer.Option(False, "--yes", "-y")):
    """Cancella TUTTE le memorie di uno scope."""
    if not yes:
        confirm = typer.confirm(f"Cancellare TUTTE le memorie di '{scope}'?")
        if not confirm:
            return
    n = MemoryManager.delete_scope(scope)
    console.print(f"[red]✗[/] {n} memorie cancellate dallo scope '{scope}'.")


@app.command("purge")
def purge(yes: bool = typer.Option(False, "--yes", "-y")):
    """Cancella TUTTE le memorie (irreversibile)."""
    if not yes:
        confirm = typer.confirm("Cancellare DEFINITIVAMENTE tutte le memorie?")
        if not confirm:
            return
    n = MemoryManager.purge_all()
    console.print(f"[red]✗[/] {n} memorie cancellate (tutto lo store).")


if __name__ == "__main__":
    app()
