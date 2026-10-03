"""Remote EmbeddingProvider — Fase 12."""
from __future__ import annotations

from ._client import request_json


class RemoteEmbeddingProvider:
    """Implementa EmbeddingProvider via HTTP POST /embed."""

    def __init__(self):
        # Lazy: chiediamo i metadata solo al primo accesso a `dimension`/`model_name`.
        self._dim: int | None = None
        self._model: str | None = None

    def _ensure_meta(self) -> None:
        if self._dim is not None and self._model is not None:
            return
        # Embed di test su una stringa fake per scoprire dim/model
        data = request_json("POST", "/embed", json_body={"texts": ["x"]})
        self._dim = int(data.get("dimension", 0))
        self._model = str(data.get("model", "remote"))

    @property
    def dimension(self) -> int:
        self._ensure_meta()
        return self._dim or 0

    @property
    def model_name(self) -> str:
        self._ensure_meta()
        return self._model or "remote"

    def embed(self, text: str) -> list[float]:
        data = request_json("POST", "/embed", json_body={"texts": [text]})
        embeddings = data.get("embeddings") or [[]]
        return embeddings[0] if embeddings else []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed in batch. Chunka in lotti di EMBED_BATCH per non saturare
        il timeout HTTP quando il server è CPU-only (e quindi lento)."""
        if not texts:
            return []
        EMBED_BATCH = 16
        all_embeddings: list[list[float]] = []
        n_total = len(texts)
        for i in range(0, n_total, EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            data = request_json("POST", "/embed", json_body={"texts": batch})
            embs = data.get("embeddings") or []
            all_embeddings.extend(embs)
            if n_total > EMBED_BATCH:
                print(f"  [emb] {min(i + EMBED_BATCH, n_total)}/{n_total} embed completati", flush=True)
        return all_embeddings
