#!/usr/bin/env python3
"""Benchmark embeddings server-side: bge-m3 su CPU vs latency budget L1.

Risponde a OQ-16 (DESIGN_THIN_CLIENT.md §11): il watcher L1 fa una similarity
per ogni utterance final (~1 embed di frase ogni 2-5s di call). Su Fly.io
shared-cpu il modello locale può essere il collo di bottiglia — questo script
misura la latenza reale per batch tipici.

Da eseguire: (a) sul Mac di Matteo con `taskset`/limite CPU per simulare Fly,
oppure (b) direttamente su una VM Fly di test (più fedele).

Uso:
    python scripts/bench_embeddings.py            # usa il provider local del repo
    python scripts/bench_embeddings.py --n 50     # più iterazioni

Soglia decisionale (dal latency budget §5): L1 totale < 200ms → embed di una
frase deve stare sotto ~100ms p95. Sopra → passare a embeddings API (Voyage).
"""
from __future__ import annotations

import argparse
import statistics
import time

SAMPLES = [
    "quanto costa il piano pro per il team di vendita?",
    "possiamo integrare il vostro sistema con Salesforce e HubSpot senza sviluppo custom?",
    "il cliente ha chiesto se la clausola di recesso vale anche per il primo trimestre",
    "what is your pricing for enterprise deployments with SSO?",
    "ci sono referenze nel settore energy per questo tipo di implementazione?",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20, help="iterazioni per campione")
    args = ap.parse_args()

    # Forza provider local (bge-m3) come farebbe il server
    import meeting_copilot.config as cfg
    from dataclasses import replace
    cfg.PROVIDERS = replace(cfg.PROVIDERS, embedding="local")
    from meeting_copilot.providers import get_embedding_provider

    prov = get_embedding_provider()
    print(f"Provider: {prov.model_name} · dim={prov.dimension}")

    # warmup (caricamento modello escluso dalle misure)
    t0 = time.monotonic()
    prov.embed_batch(SAMPLES[:1])
    print(f"Warmup (load + primo embed): {time.monotonic() - t0:.2f}s\n")

    # caso reale L1: embed di UNA frase alla volta (ogni utterance final)
    lat: list[float] = []
    for _ in range(args.n):
        for s in SAMPLES:
            t = time.monotonic()
            prov.embed_batch([s])
            lat.append((time.monotonic() - t) * 1000)

    lat.sort()
    p50 = statistics.median(lat)
    p95 = lat[int(len(lat) * 0.95)]
    print(f"Embed singola frase ({len(lat)} misure):")
    print(f"  p50: {p50:.0f}ms · p95: {p95:.0f}ms · max: {lat[-1]:.0f}ms")

    verdict = ("OK per L1 (< 100ms p95) → bge-m3 CPU regge"
               if p95 < 100 else
               "TROPPO LENTO per L1 su questa CPU → valutare Voyage/OpenAI API (OQ-16)")
    print(f"\n→ {verdict}")
    print("NB: su Fly shared-cpu aspettati 2-4x più lento di un M-series.")


if __name__ == "__main__":
    main()
