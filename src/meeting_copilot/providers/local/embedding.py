"""Local embedding provider (Ollama backend).

Supporta:
- "nomic-embed-text"      → 768-dim, multilingue OK (baseline)
- "bge-m3"                → 1024-dim, multilingue top + supporta sparse/dense/colbert
- "mxbai-embed-large"     → 1024-dim, English-focused (skip per IT)
- "snowflake-arctic-embed"→ 1024-dim, alternative

Per Fase 9.2: switch a bge-m3 → migliore qualità retrieval, soprattutto IT.
"""
from __future__ import annotations

from ...config import RAG


class LocalEmbeddingProvider:
    """Implementa EmbeddingProvider via Ollama embeddings endpoint."""

    # Dimensione di output per modello noto (per validazione, no I/O)
    _DIMENSIONS = {
        "nomic-embed-text": 768,
        "bge-m3": 1024,
        "mxbai-embed-large": 1024,
        "snowflake-arctic-embed": 1024,
    }

    def __init__(self, model: str | None = None):
        self._model = model or RAG.embedding_model
        self._dim = self._DIMENSIONS.get(self._model.split(":")[0], 1024)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model

    def embed(self, text: str) -> list[float]:
        import ollama
        resp = ollama.embeddings(model=self._model, prompt=text)
        return resp["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Usa l'API batch nativa di Ollama (`ollama.embed`) che processa
        TUTTI i testi in una singola call. Significativamente più veloce di
        un loop su `ollama.embeddings()` (single-prompt).

        Fallback a loop sequenziale se la batch API non è disponibile
        (vecchie versioni di Ollama).
        """
        import ollama
        if not texts:
            return []
        try:
            resp = ollama.embed(model=self._model, input=list(texts))
            embeddings = resp.get("embeddings") or []
            if embeddings and len(embeddings) == len(texts):
                return embeddings
            # fallback se Ollama non ha ritornato lista corretta
        except Exception as e:
            print(f"[embed_batch] batch API fallita ({e}), fallback to loop")
        return [self.embed(t) for t in texts]
