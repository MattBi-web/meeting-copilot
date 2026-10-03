"""Test retrieval pipeline da CLI (senza audio).

Uso:
    python test_retrieval.py "come funziona WebSocket con ProductA?"
    python test_retrieval.py                         # usa default query
    python test_retrieval.py --section acme_demo "altra query"

Misura embed / hybrid_retrieve / rerank separati, così vedi quale step
è il collo di bottiglia (le print di [rag] vengono dal retriever stesso).
"""
from __future__ import annotations

import argparse
import sys
import time

from meeting_copilot.kb.manager import KnowledgeManager


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?",
                    default="come funziona WebSocket con ProductA?",
                    help="Query da testare")
    ap.add_argument("--section", "-s", default=None,
                    help="Section slug (default: usa active_sections)")
    ap.add_argument("--top_k", "-k", type=int, default=4)
    ap.add_argument("--runs", "-r", type=int, default=3,
                    help="Quante chiamate ripetute (per misurare warm vs cold)")
    args = ap.parse_args()

    if args.section:
        sections = [args.section]
    else:
        sections = KnowledgeManager.get_active_sections()
    print(f"\n=== Test retrieval ===")
    print(f"Query   : {args.query!r}")
    print(f"Sections: {sections}")
    print(f"Top-k   : {args.top_k}")
    print(f"Runs    : {args.runs}\n")

    if not sections:
        print("❌ Nessuna section attiva. Usa: --section acme_demo")
        return 1

    for i in range(1, args.runs + 1):
        print(f"─── RUN {i}/{args.runs} ───")
        t0 = time.monotonic()
        try:
            hits = KnowledgeManager.search(
                args.query, sections=sections, top_k=args.top_k
            )
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return 2
        dt = time.monotonic() - t0
        print(f"  → TOTALE: {dt*1000:.0f}ms, hits={len(hits)}")
        if hits:
            for j, h in enumerate(hits[:args.top_k], 1):
                src = getattr(h, "source", "?")
                sec = getattr(h, "section", "?")
                score = getattr(h, "score", 0.0)
                snippet = (getattr(h, "text", "") or "")[:80].replace("\n", " ")
                print(f"   [{j}] score={score:.3f} sec={sec} src={src}")
                print(f"       {snippet}…")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
